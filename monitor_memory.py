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

CPU memory comes from /proc/<pid>/status: VmRSS sampled over time, and VmHWM,
which is the kernel's own high-water mark and so is exact even if the true peak
falls between two samples.

Usage:
    ./monitor_memory.py --out mem.csv --summary mem_summary.txt
    ./monitor_memory.py --out mem.csv --comm stereo_inertial --interval 0.05
"""

import argparse
import ctypes
import os
import sys
import time

try:
    import pynvml
except ImportError:
    pynvml = None

# Linux truncates /proc/<pid>/comm to 15 characters, so the process to look for is
# "stereo_inertial", never the full "stereo_inertial_tum_vi".
DEFAULT_COMM = 'stereo_inertial'
MIB = 1024 * 1024


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


class Gpu:
    """NVML device queries, degrading to zeros if NVML is unavailable."""

    def __init__(self, index=0):
        self.handle = None
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as e:                                # NVML absent or no device
            print(f'[monitor] NVML unavailable: {e}', file=sys.stderr)
            self.handle = None

    def device_used(self):
        if self.handle is None:
            return 0.0
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / MIB
        except Exception:
            return 0.0

    def process_used(self, pid):
        """MiB charged to `pid`, or None when NVML does not list it (PID namespace
        mismatch in a container, or the process holds no CUDA context yet)."""
        if self.handle is None:
            return None
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
    args = ap.parse_args()

    gpu = Gpu(args.gpu)

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
            peak_dev = max(peak_dev, dev)
            peak_rss = max(peak_rss, rss)
            hwm = max(hwm, st.get('VmHWM', 0.0))

            f.write(f'{t:.3f},{"" if pu is None else f"{pu:.1f}"},{dev:.1f},'
                    f'{dev - baseline:.1f},{rss:.1f},{st.get("VmSize", 0.0):.1f},'
                    f'{st.get("Threads", 0)}\n')
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
        f'gpu_baseline_mib     {baseline:.1f}',
        f'gpu_peak_process_mib {peak_proc:.1f}' if proc_seen else
        'gpu_peak_process_mib n/a (NVML did not list this PID; use device delta)',
        f'gpu_peak_device_mib  {peak_dev:.1f}',
        f'gpu_peak_delta_mib   {peak_dev - baseline:.1f}',
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
