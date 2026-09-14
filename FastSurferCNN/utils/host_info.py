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
of the same code on two hosts can differ in the last bits. Recording the host makes that
visible in the log instead of leaving it to be guessed afterwards.

Run as a module to print the block for a shell log. It has to be `-m`: running the file
directly puts `FastSurferCNN/utils` on the path, where `logging.py` shadows the standard
library module that torch needs.

    python3 -m FastSurferCNN.utils.host_info
"""

import argparse
import os
import platform
import re
import subprocess

__all__ = [
    "cpu_model",
    "host_info",
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


def _cpu_count() -> str:
    """Report the cores the process may actually use, and the cores the host has."""
    total = os.cpu_count()
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # sched_getaffinity is linux only
        available = total
    if available == total:
        return f"{available}"
    return f"{available} of {total}"


def torch_info(with_threads: bool = True) -> list[str]:
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

    Returns
    -------
    list[str]
        One line per fact, ready to be written to a log.
    """
    try:
        import torch
    except ImportError as e:
        return [f"Torch: not importable ({e})"]

    # the vector ISA torch selected, e.g. AVX2 or AVX512, which decides the kernel and
    # therefore the last bits of every reduction
    get_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    capability = get_capability() if get_capability else "unknown"
    line = f"Torch {torch.__version__}, CPU capability {capability}"
    if with_threads:
        line += f", {torch.get_num_threads()} intra-op and {torch.get_num_interop_threads()} inter-op threads"
    lines = [line]
    if torch.cuda.is_available():
        devices = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        lines.append(f"CUDA {torch.version.cuda}, devices: {devices}")
    return lines


def host_info(with_torch: bool = False) -> list[str]:
    """
    Describe the machine this process is running on.

    Parameters
    ----------
    with_torch : bool, default=False
        Whether to import torch and report the kernels it selected. Off by default
        because the caller is usually a log header rather than the process doing the
        work, so only the build facts would be true; the thread counts are left out for
        the same reason. A step that runs torch itself should call `torch_info` once its
        thread count and device are final.

    Returns
    -------
    list[str]
        One line per fact, ready to be written to a log.
    """
    lines = [
        f"Host: {platform.node()}",
        f"Platform: {platform.system()} {platform.release()} {platform.machine()}",
        f"CPU: {cpu_model()}",
        f"CPU cores: {_cpu_count()}",
    ]
    if with_torch:
        lines += torch_info(with_threads=False)
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
    args = parser.parse_args()
    print("\n".join(host_info(with_torch=args.torch)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
