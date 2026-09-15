# Copyright 2026 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Describe the machine a run executed on.

Floating point results depend on which vectorised kernels the CPU supports, so two runs
of the same code on two hosts can differ in the last bits. Recording the machine makes
that visible in the log instead of leaving it to be guessed afterwards.

The hostname is deliberately absent: `uname -a` in recon-surf.log, `Machine:` in
recon-all.log, `HOSTNAME` in recon-all.env and `HOST` in recon-surf.done already carry
it, and inside a container it is a random string that identifies nothing.

Run as a script to print the block for a shell log:

    python3 FastSurferCNN/host_info.py
"""

import argparse
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger

__all__ = [
    "cpu_count",
    "cpu_model",
    "cpu_quota",
    "host_info",
    "log_torch_info",
    "numerical_fingerprint",
    "torch_info",
]

# thread limits that change the reduction order, and so the last bits, of the result
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)

CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_CPU_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_CPU_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def cpu_model() -> str:
    """
    Get the marketing name of the CPU.

    Returns
    -------
    str
        The CPU model, or "unknown" if the platform does not report one.
    """
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                match = re.match(r"^model name\s*:\s*(.+)$", line)
                if match:
                    return match.group(1).strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown"


def cpu_quota() -> float | None:
    """
    Get the CPU limit the cgroup imposes, in cores.

    `docker run --cpus=2` sets a quota rather than an affinity mask, so the core count
    alone does not describe what the process may use.

    Returns
    -------
    float, None
        The limit in cores, or None if the process is not capped.
    """
    try:
        quota, period = CGROUP_V2_CPU_MAX.read_text().split()
        if quota == "max":
            return None
        if int(period) > 0:
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int(CGROUP_V1_CPU_QUOTA.read_text())
        period = int(CGROUP_V1_CPU_PERIOD.read_text())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def cpu_count() -> str:
    """
    Describe how many cores the process may use.

    Returns
    -------
    str
        The usable core count, qualified by the host total and by a cgroup limit where
        those differ from it.
    """
    total = os.cpu_count()
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # sched_getaffinity is linux only
        available = total
    parts = ["unknown" if available is None else str(available)]
    if total is not None and available != total:
        parts.append(f"of {total}")
    quota = cpu_quota()
    if quota is not None:
        parts.append(f"capped at {quota:g} by the cgroup")
    return " ".join(parts)


def numerical_fingerprint() -> str:
    """
    Hash the result of a convolution and a softmax, to identify what this host computes.

    The CPU model does not predict the arithmetic: the same model appears with and
    without AVX512 depending on what the hypervisor exposes, and two hosts reporting the
    same capability can still differ. This measures the answer instead of describing the
    machine, so two logs can be compared by one line.

    Convolution and softmax because those are what the networks use. Matmul is left out:
    it is vendor dependent through BLAS, but no model here calls it.

    Returns
    -------
    str
        `conv=<hash> soft=<hash>`, or a note if torch is missing.
    """
    import hashlib
    import struct

    try:
        import torch
    except ImportError as e:
        return f"not available ({type(e).__name__})"

    def digest(t: "torch.Tensor") -> str:
        # via int32 rather than numpy, so a missing numpy cannot break the log line
        values = t.detach().contiguous().flatten().view(torch.int32).tolist()
        return hashlib.md5(struct.pack(f"<{len(values)}i", *values)).hexdigest()[:12]

    # Integer arithmetic then an inexact divide. Elementwise division is correctly
    # rounded, so the inputs are identical on every host, while the values are not
    # exactly representable, which is what lets a reordered reduction show.
    n = 1 * 16 * 64 * 64
    x = (((torch.arange(n, dtype=torch.float32) % 257) - 128) / 3.0).reshape(1, 16, 64, 64)
    w = (((torch.arange(32 * 16 * 3 * 3, dtype=torch.float32) % 97) - 48) / 7.0).reshape(32, 16, 3, 3)
    with torch.no_grad():
        conv = torch.nn.functional.conv2d(x, w, padding=1)
        soft = torch.softmax(x.reshape(-1), 0)
    return f"conv={digest(conv)} soft={digest(soft)}"


def torch_info(with_threads: bool = True, with_fingerprint: bool = False) -> list[str]:
    """
    Describe what torch will dispatch to on this host.

    Call this after the thread count and the device have been set, so the reported
    values are the ones the run actually used.

    Parameters
    ----------
    with_threads : bool, default=True
        Whether to report the thread counts. Pass False from a process that does no
        torch work itself, such as a shell log header, where the counts would be torch's
        defaults rather than anything the run will use.
    with_fingerprint : bool, default=False
        Whether to add `numerical_fingerprint`. Costs a few milliseconds, so it is off
        for callers that only want the build facts.

    Returns
    -------
    list[str]
        One line per fact, ready to be written to a log.
    """
    try:
        import torch
    except ImportError as e:
        return [f"Torch: not importable ({type(e).__name__}: {str(e).splitlines()[0]})"]

    # the vector ISA torch selected, e.g. AVX2 or AVX512, which decides the kernel and
    # therefore the last bits of every reduction
    line = f"Torch {torch.__version__}, CPU capability {torch.backends.cpu.get_cpu_capability()}"
    if with_threads:
        line += f", {torch.get_num_threads()} intra-op and {torch.get_num_interop_threads()} inter-op threads"
    lines = [line]
    if torch.cuda.is_available():
        devices = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        lines.append(f"CUDA {torch.version.cuda}, devices: {devices}")
    if with_fingerprint:
        lines.append(f"Numerical fingerprint: {numerical_fingerprint()}")
    return lines


def log_torch_info(logger: "Logger", with_fingerprint: bool = False) -> None:
    """
    Log what torch dispatched to, one record per line.

    Call this after the thread count and the device have been set, so the reported
    values are the ones the run actually used.

    Parameters
    ----------
    logger : logging.Logger
        The logger to write to.
    with_fingerprint : bool, default=False
        Whether to include `numerical_fingerprint`. Off by default: the fingerprint
        describes the host, which does not change during a run, so the log header records
        it once rather than every network repeating it. The thread counts on the line
        above do differ per process, which is why those are reported here.
    """
    for line in torch_info(with_fingerprint=with_fingerprint):
        # attribute the record to the caller, so the log says which network emitted it
        logger.info(line, stacklevel=2)


def host_info(with_torch: bool = False, with_fingerprint: bool = False) -> list[str]:
    """
    Describe the machine this process is running on.

    Parameters
    ----------
    with_torch : bool, default=False
        Whether to import torch and report the kernels it selected. Off by default
        because the caller is usually a log header rather than the process doing the
        work, so only the build facts would be true; the thread counts are left out for
        the same reason. A step that runs torch itself should call `log_torch_info` once
        its thread count and device are final.
    with_fingerprint : bool, default=False
        Whether to add `numerical_fingerprint`. Implies `with_torch`.

    Returns
    -------
    list[str]
        One line per fact, ready to be written to a log.
    """
    lines = [
        f"Platform: {platform.system()} {platform.release()} {platform.machine()}",
        f"CPU: {cpu_model()}",
        f"CPU cores: {cpu_count()}",
    ]
    if with_torch or with_fingerprint:
        lines += torch_info(with_threads=False, with_fingerprint=with_fingerprint)
    limits = [f"{var}={os.environ[var]}" for var in THREAD_VARS if var in os.environ]
    lines.append(f"Thread limits: {', '.join(limits) if limits else 'none set'}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a description of the machine this runs on, for the log.")
    parser.add_argument(
        "--torch",
        action="store_true",
        help="also report the torch build, for a step that runs torch but does not log this itself",
    )
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="also hash a convolution and a softmax, so two logs can be compared for whether the "
             "hosts computed the same thing at all. Implies --torch.",
    )
    args = parser.parse_args()
    print("\n".join(host_info(with_torch=args.torch, with_fingerprint=args.fingerprint)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
