"""TorchScript helpers for CPU inference hot paths."""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable

import torch


def env_flag_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def should_trace_cpu_inference(
    *,
    out_scale: object,
    device: torch.device,
    batch_size: int,
    env_var: str,
) -> bool:
    return (
        out_scale is None
        and device.type == "cpu"
        and batch_size == 1
        and env_flag_enabled(env_var)
    )


def trace_for_inference(
    *,
    model: torch.nn.Module,
    wrapper_factory: Callable[[torch.nn.Module], torch.nn.Module],
    example_inputs: tuple[torch.Tensor, ...],
    freeze: bool,
    logger,
    label: str,
) -> torch.nn.Module:
    trace_start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        traced_model = torch.jit.trace(
            wrapper_factory(model),
            example_inputs,
            check_trace=False,
        )
        traced_model.eval()
        if freeze:
            traced_model = torch.jit.freeze(traced_model)
    logger.info(f"Traced {label} model in {time.time() - trace_start:0.4f} seconds")
    return traced_model
