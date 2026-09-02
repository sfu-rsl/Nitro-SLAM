#!/usr/bin/env python3
"""Sample board power, CPU and GPU utilisation of a SLAM run into a time series.

Companion to monitor_memory.py, same shape: starts before the binary, waits for it
to appear, polls until it exits, writes one CSV row per sample plus a summary.

Everything comes from the jtop daemon, which reads the Orin's INA3221 rails and the
tegra sysfs nodes as root and serves any client in the `jtop` group, so the sampler
needs no privileges of its own.

  * power   VDD_IN is the board total; VDD_CPU_GPU_CV and VDD_SOC are the two rails
            that sum to most of it. All in milliwatts, as reported -- the INA3221
            integrates over its own conversion window, so these are already averaged
            over roughly a millisecond and are not instantaneous.
  * gpu     load is the tegra 3d-scaling busy percentage, not an occupancy figure:
            it says the GPU had work, not how well the work filled it.
  * cpu     per-core busy = 100 - idle, from the same jiffy counters top reads, so a
            core parked in WFI reads 0 rather than its floor. cpu_total_pct is the mean
            of those cores, not jtop's own `cpu.total`, which is windowed differently
            and lags the per-core series by about a second.
  * proc    the run's own CPU share and RSS, from /proc/<pid>/stat and /status, so a
            spike in board power can be attributed to the run rather than to a
            neighbour.

The GPU and power columns only change when the daemon broadcasts, so --interval
below the daemon period yields repeated values, not finer resolution.

With --events, marker lines from the run's own stdout are timestamped into a second
CSV, which is what makes a loop closure locatable in the power trace. The run's
stdout is block-buffered into ostream.txt by the shell redirect, so a marker's
recorded time lags its real time by however long the 4 KiB buffer takes to fill --
typically well under a second while LIBA lines are flowing, but do not read these
timestamps as exact.

Usage:
    ./monitor_power.py --out power.csv --summary power_summary.txt
    ./monitor_power.py --out power.csv --events events.csv --ostream run/ostream.txt
"""

import argparse
import os
import re
import sys
import time

try:
    from jtop import jtop
except ImportError:
    jtop = None

DEFAULT_COMM = 'stereo_inertial'
CLK_TCK = os.sysconf('SC_CLK_TCK')

# Lines worth a timestamp. Loop closure and inertial initialisation are the two
# events that show up as a power excursion; the rest bracket the run.
MARKERS = [
    ('loop_found',      re.compile(r'Good loop found!')),
    ('loop_detected',   re.compile(r'PR: Loop detected')),
    ('loop_bad',        re.compile(r'BAD LOOP')),
    ('merge_check',     re.compile(r'Merge check transformation')),
    ('viba_start',      re.compile(r'start VIBA (\d)')),
    ('viba_end',        re.compile(r'end VIBA (\d)')),
    ('map_created',     re.compile(r'New Map created')),
    ('map_reset',       re.compile(r'LM: Reseting current map')),
    ('tracked_all',     re.compile(r'Tracked \d+ images')),
    ('shutdown',        re.compile(r'^Shutdown')),
]


def find_pid(comm, exclude):
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in exclude:
            continue
        try:
            with open(f'/proc/{pid}/comm') as fh:
                if fh.read().strip() == comm:
                    return pid
        except OSError:
            continue
    return None


def proc_cpu_jiffies(pid):
    """(utime + stime) in jiffies, or None once the process is gone."""
    try:
        with open(f'/proc/{pid}/stat') as fh:
            fields = fh.read().rsplit(') ', 1)[1].split()
        return int(fields[11]) + int(fields[12])   # utime, stime
    except (OSError, IndexError, ValueError):
        return None


def proc_rss_mib(pid):
    try:
        with open(f'/proc/{pid}/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


class OstreamTail:
    """Follow a growing file and yield (marker, text) for lines that match."""

    def __init__(self, path):
        self.path = path
        self._fh = None
        self._buf = ''

    def poll(self):
        if self._fh is None:
            if not os.path.exists(self.path):
                return
            try:
                self._fh = open(self.path, 'r', errors='ignore')
            except OSError:
                return
        chunk = self._fh.read()
        if not chunk:
            return
        self._buf += chunk
        *lines, self._buf = self._buf.split('\n')
        for line in lines:
            for name, pat in MARKERS:
                if pat.search(line):
                    yield name, line.strip()
                    break

    def close(self):
        if self._fh is not None:
            self._fh.close()


def fmt(x, nd=1):
    return '' if x is None else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='CSV path for the time series')
    ap.add_argument('--summary', help='text path for the summary')
    ap.add_argument('--events', help='CSV path for timestamped stdout markers')
    ap.add_argument('--ostream', help='the run ostream.txt to follow for --events')
    ap.add_argument('--comm', default=DEFAULT_COMM, help='process comm to attach to')
    ap.add_argument('--interval', type=float, default=0.1, help='sample period in seconds')
    ap.add_argument('--wait', type=float, default=300.0,
                    help='seconds to wait for the process to appear')
    args = ap.parse_args()

    if jtop is None:
        print('[power] jetson-stats not installed (pip install jetson-stats)', file=sys.stderr)
        return 1

    ncpu = None
    rows = []
    events = []
    tail = OstreamTail(args.ostream) if (args.events and args.ostream) else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with jtop(interval=args.interval) as j:
        if not j.ok():
            print('[power] jtop unavailable; is jtop.service running and are you in '
                  'the "jtop" group?', file=sys.stderr)
            return 1

        # Baseline: a couple of seconds before the run starts. This is whatever the
        # board was already doing, not a true idle floor -- the sampler and the shell
        # that launched it are both in it -- so read it as "what this run added to",
        # not as the Orin's resting draw.
        base = []
        t_base = time.time()
        while time.time() - t_base < 2.0:
            if j.ok():
                base.append(j.power['tot']['power'])
            time.sleep(args.interval)
        idle_mw = sum(base) / len(base) if base else None

        deadline = time.time() + args.wait
        pid = None
        while time.time() < deadline:
            pid = find_pid(args.comm, exclude={os.getpid()})
            if pid is not None:
                break
            time.sleep(0.02)
        if pid is None:
            print(f'[power] no process named "{args.comm}" appeared within {args.wait}s',
                  file=sys.stderr)
            return 1

        t0 = time.time()
        last_j, last_t = proc_cpu_jiffies(pid), t0
        fh = open(args.out, 'w', buffering=1)
        header_written = False

        while j.ok():
            now = time.time()
            cur_j = proc_cpu_jiffies(pid)
            if cur_j is None:                     # process gone
                break

            p = j.power
            g = j.gpu['gpu']
            cpus = j.cpu['cpu']
            if ncpu is None:
                ncpu = len(cpus)
                fh.write('t_s,wall_unix,power_in_mw,power_cpu_gpu_cv_mw,power_soc_mw,'
                         'gpu_load_pct,gpu_freq_khz,cpu_total_pct,'
                         + ','.join(f'cpu{i}_pct' for i in range(ncpu)) + ','
                         + 'cpu_freq_khz,temp_cpu_c,temp_gpu_c,temp_soc_c,'
                           'proc_cpu_pct,proc_rss_mib,ram_used_mib\n')
                header_written = True

            dt = now - last_t
            proc_pct = ((cur_j - last_j) / CLK_TCK / dt * 100.0) if dt > 0 else None
            last_j, last_t = cur_j, now

            tot_mw = p['tot']['power']
            rail = p['rail']
            cg = rail.get('VDD_CPU_GPU_CV', {}).get('power')
            soc = rail.get('VDD_SOC', {}).get('power')
            gpu_load = g['status']['load']
            gpu_freq = g['freq']['cur']
            core_pct = [100.0 - c.get('idle', 100.0) for c in cpus]
            cpu_total = sum(core_pct) / len(core_pct)
            cpu_freq = sum(c['freq']['cur'] for c in cpus) / len(cpus)
            temps = j.temperature
            t_cpu = temps.get('cpu', {}).get('temp')
            t_gpu = temps.get('gpu', {}).get('temp')
            t_soc = temps.get('soc0', {}).get('temp')
            rss = proc_rss_mib(pid)
            ram = j.memory['RAM']['used'] / 1024.0

            el = now - t0
            fh.write(f'{el:.3f},{now:.3f},{fmt(tot_mw,0)},{fmt(cg,0)},{fmt(soc,0)},'
                     f'{fmt(gpu_load)},{fmt(gpu_freq,0)},{fmt(cpu_total)},'
                     + ','.join(fmt(c) for c in core_pct) + ','
                     + f'{fmt(cpu_freq,0)},{fmt(t_cpu)},{fmt(t_gpu)},{fmt(t_soc)},'
                       f'{fmt(proc_pct)},{fmt(rss)},{fmt(ram,0)}\n')
            rows.append((el, tot_mw, cg, soc, gpu_load, cpu_total, proc_pct, t_cpu, t_gpu))

            if tail is not None:
                for name, text in tail.poll():
                    events.append((el, now, name, text))

            time.sleep(args.interval)

        fh.close()
        if tail is not None:
            for name, text in tail.poll():           # drain whatever landed last
                events.append((time.time() - t0, time.time(), name, text))
            tail.close()

    if args.events:
        with open(args.events, 'w') as fh:
            fh.write('t_s,wall_unix,marker,line\n')
            for el, wall, name, text in events:
                fh.write(f'{el:.3f},{wall:.3f},{name},"{text}"\n')

    if args.summary and rows:
        dur = rows[-1][0]
        def col(i): return [r[i] for r in rows if r[i] is not None]
        # Trapezoidal energy over the sampled series; the last sample carries no
        # interval of its own, so it contributes nothing.
        energy_j = 0.0
        for a, b in zip(rows, rows[1:]):
            if a[1] is not None and b[1] is not None:
                energy_j += (a[1] + b[1]) / 2.0 / 1000.0 * (b[0] - a[0])
        p_in = col(1)
        with open(args.summary, 'w') as fh:
            fh.write(f'samples              {len(rows)}\n')
            fh.write(f'duration_s           {dur:.1f}\n')
            fh.write(f'interval_s           {args.interval}\n')
            fh.write(f'baseline_power_mw        {fmt(idle_mw,0)}\n')
            fh.write(f'power_mean_mw        {sum(p_in)/len(p_in):.0f}\n')
            fh.write(f'power_peak_mw        {max(p_in):.0f}\n')
            if idle_mw is not None:
                fh.write(f'power_above_baseline_mw  {sum(p_in)/len(p_in) - idle_mw:.0f}\n')
            fh.write(f'energy_j             {energy_j:.1f}\n')
            for name, i in [('cpu_gpu_cv_mean_mw', 2), ('soc_mean_mw', 3),
                            ('gpu_load_mean_pct', 4), ('cpu_total_mean_pct', 5),
                            ('proc_cpu_mean_pct', 6)]:
                v = col(i)
                if v:
                    fh.write(f'{name:<20} {sum(v)/len(v):.1f}\n')
            for name, i in [('gpu_load_peak_pct', 4), ('cpu_total_peak_pct', 5),
                            ('temp_cpu_peak_c', 7), ('temp_gpu_peak_c', 8)]:
                v = col(i)
                if v:
                    fh.write(f'{name:<20} {max(v):.1f}\n')
            fh.write(f'events               {len(events)}\n')

    print(f'[power] {len(rows)} samples, {len(events)} events', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
