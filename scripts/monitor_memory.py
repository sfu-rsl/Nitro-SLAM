#!/usr/bin/env python3
"""Sample GPU and CPU memory of a SLAM run and write a per-run time series.

Runs as a separate process alongside the binary: it waits for the process to
appear, polls until it exits, and writes one CSV row per sample plus a summary
of the peaks.

GPU memory is read two ways, because neither is reliable on its own:

  * per-process, via NVML's compute-process list. This is the number that
    actually belongs to the run -- but NVML reports host PIDs, so inside a
    container it may never match ours.
  * device-wide minus a baseline captured before the process started. This
    always works, but any other GPU tenant lands in it.

Both are recorded; `--summary` reports which one was usable.

On Tegra (Jetson) NVML loads but answers `Not Supported` to every memory query:
the iGPU has no VRAM of its own, it allocates from system RAM. There the numbers
come from the jtop daemon instead, which reads the same two figures out of
/sys/kernel/debug/nvmap -- root-only, but the daemon runs as root and answers any
client in the `jtop` group, so the sampler does not need privileges of its own.
Both columns keep their meaning across the two backends: gpu_proc is the run's own
GPU memory, gpu_dev_used is the whole device's. A desktop CSV and a Jetson CSV can
therefore be read the same way.

/proc/meminfo is deliberately not used as a stand-in for any of this. The nvmap
driver keeps freed pages in a pool of its own, so a repeat run is served from the
pool without the kernel counters moving: measured against a known workload it
under-reported a 2 GiB run as 1.3 GiB, and a 1 GiB run by as much as 6x. It fails
quietly and low, which is worse than not measuring at all.

CPU memory comes from /proc/<pid>/status: VmRSS sampled over time, and VmHWM,
which is the kernel's own high-water mark and so is exact even if the true peak
falls between two samples.

Usage:
    ./monitor_memory.py --out mem.csv --summary mem_summary.txt
    ./monitor_memory.py --out mem.csv --comm stereo_inertial --interval 0.05
"""

import argparse
import os
import sys
import time

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    from jtop import jtop
except ImportError:
    jtop = None

# Linux truncates /proc/<pid>/comm to 15 characters, so the process to look for is
# "stereo_inertial", never the full "stereo_inertial_tum_vi".
DEFAULT_COMM = 'stereo_inertial'
MIB = 1024 * 1024
KIB = 1024                    # jtop reports memory in kB


def find_pid(comm, exclude):
    """PID of the running process whose comm matches exactly, or None."""
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in exclude:
            continue
        try:
            with open(f'/proc/{pid}/comm') as f:
                if f.read().strip() == comm:
                    return pid
        except OSError:
            continue
    return None


def proc_status(pid):
    """{VmRSS, VmHWM, VmSize} in MiB plus thread count, or None once the process is gone."""
    try:
        with open(f'/proc/{pid}/status') as f:
            out = {}
            for line in f:
                k, _, v = line.partition(':')
                if k in ('VmRSS', 'VmHWM', 'VmSize'):
                    out[k] = int(v.split()[0]) / 1024        # kB -> MiB
                elif k == 'Threads':
                    out['Threads'] = int(v)
            return out
    except (OSError, ValueError, IndexError):
        return None


class NvmlGpu:
    """Per-process and device-wide GPU memory from NVML. Discrete GPUs only."""

    name = 'NVML'

    def __init__(self, index=0):
        self.handle = None
        self.supported = False
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as e:                                # NVML absent or no device
            print(f'[monitor] NVML unavailable: {e}', file=sys.stderr)
            self.handle = None
            return
        # Probe once at startup instead of swallowing the same error every sample.
        # On Tegra nvmlInit and the handle both succeed and only the memory query
        # fails, so loading NVML is not on its own evidence that this will work.
        try:
            pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.supported = True
        except Exception as e:
            print(f'[monitor] NVML reports no device memory ({e}); '
                  f'expected on Jetson/Tegra, where the iGPU allocates from system RAM',
                  file=sys.stderr)

    def device_used(self):
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / MIB
        except Exception:
            return None

    def process_used(self, pid):
        """MiB charged to `pid`, or None when NVML does not list it -- a PID namespace
        mismatch in a container, or no CUDA context yet."""
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except Exception:
            return None
        for p in procs:
            if p.pid == pid:
                used = getattr(p, 'usedGpuMemory', None)
                return None if used is None else used / MIB
        return None

    def shutdown(self):
        if self.handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class JtopGpu:
    """Per-process and device-wide GPU memory on Tegra, via the jtop daemon.

    The underlying source is /sys/kernel/debug/nvmap, which is root-only; the jtop
    service already runs as root and serves any client in the `jtop` group, so this
    works unprivileged where reading debugfs directly would not.

    Device-wide is RAM.shared, the total handed out to the GPU, which measures
    allocations rather than kernel free pages and so returns cleanly to baseline
    when a run exits -- the nvmap page pool does not distort it the way it distorts
    /proc/meminfo. Per-process is that same table split by PID.

    The daemon broadcasts on its own clock, so readings are held between updates
    rather than resampled per call; `interval` sets that rate.
    """

    name = 'jtop'

    def __init__(self, interval=0.5):
        self._j = None
        self.supported = False
        if jtop is None:
            print('[monitor] jetson-stats not installed; no GPU memory source on this '
                  'board (pip install jetson-stats)', file=sys.stderr)
            return
        try:
            self._j = jtop(interval=interval)
            self._j.start()
            # The daemon needs a broadcast to land before any field is populated.
            deadline = time.time() + 5.0
            while time.time() < deadline and not self._j.memory:
                time.sleep(0.05)
            if not self._j.memory:
                raise RuntimeError('no data broadcast within 5s')
            self.supported = True
        except Exception as e:
            print(f'[monitor] jtop unavailable ({e}); is jtop.service running, and is '
                  f'this user in the "jtop" group?', file=sys.stderr)
            self.shutdown()
            self._j = None

    def device_used(self):
        try:
            return self._j.memory['RAM']['shared'] / KIB
        except Exception:
            return None

    def process_used(self, pid):
        """MiB charged to `pid`, or None until it holds a GPU allocation.

        jtop process rows are [pid, user, gpu, type, prio, state, cpu, mem, gpu_mem,
        name] with gpu_mem in kB.
        """
        try:
            for p in self._j.processes:
                if p[0] == pid:
                    return p[8] / KIB
        except Exception:
            return None
        return None

    def shutdown(self):
        if self._j is not None:
            try:
                self._j.close()
            except Exception:
                pass


class NoGpu:
    """Stand-in when no backend works, so the sampler still records the sys_* and
    CPU columns instead of failing the run."""

    name = 'none'
    supported = False

    def device_used(self):
        return None

    def process_used(self, pid):
        return None

    def shutdown(self):
        pass


def open_gpu(index=0, interval=0.5):
    """Best available GPU memory source: NVML on a discrete GPU, jtop on Tegra."""
    g = NvmlGpu(index)
    if g.supported:
        return g
    g.shutdown()
    g = JtopGpu(interval)
    if g.supported:
        return g
    return NoGpu()


def fmt(x, nd=1):
    """CSV cell: empty for an unavailable reading, so it is never confused with 0."""
    return '' if x is None else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='CSV path for the time series')
    ap.add_argument('--summary', help='text path for the peak summary')
    ap.add_argument('--comm', default=DEFAULT_COMM, help='process comm to attach to')
    ap.add_argument('--interval', type=float, default=0.05, help='sample period in seconds')
    ap.add_argument('--wait', type=float, default=300.0,
                    help='seconds to wait for the process to appear')
    ap.add_argument('--gpu', type=int, default=0, help='NVML device index')
    ap.add_argument('--gpu-interval', type=float, default=0.5,
                    help='jtop broadcast period in seconds (Tegra only). The GPU '
                         'columns hold their value between broadcasts, so this is '
                         'their real resolution, independent of --interval.')
    args = ap.parse_args()

    gpu = open_gpu(args.gpu, args.gpu_interval)
    print(f'[monitor] gpu memory source: {gpu.name}', file=sys.stderr)

    # Baseline before the run starts, so the device-wide series can be differenced
    # against whatever was already resident on the GPU.
    baseline = gpu.device_used()

    deadline = time.time() + args.wait
    pid = None
    while time.time() < deadline:
        pid = find_pid(args.comm, exclude={os.getpid()})
        if pid is not None:
            break
        time.sleep(0.02)
    if pid is None:
        print(f'[monitor] no process named "{args.comm}" appeared within {args.wait}s',
              file=sys.stderr)
        gpu.shutdown()
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    peak_proc = 0.0
    peak_dev = baseline
    peak_rss = 0.0
    hwm = 0.0
    proc_seen = False
    t0 = time.time()

    # Line buffered: if the run is killed part-way (e.g. by `timeout`), the
    # samples taken so far are already on disk rather than lost in the buffer.
    with open(args.out, 'w', buffering=1) as f:
        f.write('t_s,gpu_proc_mib,gpu_dev_used_mib,gpu_dev_delta_mib,'
                'cpu_rss_mib,cpu_vms_mib,threads\n')
        while True:
            st = proc_status(pid)
            if st is None:
                break                                        # process exited
            dev = gpu.device_used()
            pu = gpu.process_used(pid)
            t = time.time() - t0
            rss = st.get('VmRSS', 0.0)

            if pu is not None:
                proc_seen = True
                peak_proc = max(peak_proc, pu)
            if dev is not None:
                peak_dev = dev if peak_dev is None else max(peak_dev, dev)
            peak_rss = max(peak_rss, rss)
            hwm = max(hwm, st.get('VmHWM', 0.0))

            dev_delta = None if (dev is None or baseline is None) else dev - baseline

            f.write(f'{t:.3f},{fmt(pu)},{fmt(dev)},{fmt(dev_delta)},'
                    f'{rss:.1f},{st.get("VmSize", 0.0):.1f},{st.get("Threads", 0)}\n')
            rows.append(t)
            time.sleep(args.interval)

    gpu.shutdown()

    # VmHWM is the kernel's own high-water mark, so it is the authority on peak RSS;
    # the sampled max can only ever be <= it.
    lines = [
        f'pid                  {pid}',
        f'samples              {len(rows)}',
        f'duration_s           {rows[-1]:.1f}' if rows else 'duration_s           0.0',
        f'interval_s           {args.interval}',
    ]
    lines.append(f'gpu_source           {gpu.name}')
    if gpu.supported and baseline is not None:
        lines += [
            f'gpu_baseline_mib     {baseline:.1f}',
            f'gpu_peak_process_mib {peak_proc:.1f}' if proc_seen else
            f'gpu_peak_process_mib n/a ({gpu.name} did not list this PID; use device delta)',
            f'gpu_peak_device_mib  {peak_dev:.1f}',
            f'gpu_peak_delta_mib   {peak_dev - baseline:.1f}',
        ]
    else:
        lines += [
            'gpu_peak_process_mib n/a (no GPU memory source)',
            'gpu_peak_device_mib  n/a',
            'gpu_peak_delta_mib   n/a',
        ]
    lines += [
        f'cpu_peak_rss_mib     {peak_rss:.1f}',
        f'cpu_hwm_rss_mib      {hwm:.1f}',
    ]
    text = '\n'.join(lines) + '\n'
    if args.summary:
        with open(args.summary, 'w') as f:
            f.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main())
