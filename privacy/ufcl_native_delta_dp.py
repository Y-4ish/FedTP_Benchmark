"""Native FATE aggregation adapter for UFCL update-level DP.

UFCL's historical Plain baseline is trained by ``HomoTrainerClient`` and
``AggregatorClientWrapper``.  Re-implementing that loop in a separate trainer
changes the HuggingFace trainer/replay/teacher lifecycle even when sigma is
zero.  This adapter keeps the native lifecycle intact and replaces only the
payload seen by FATE's secure aggregator:

    local model -> local delta -> clip + Gaussian noise -> native aggregation
    aggregated protected delta -> add the reference model back.

UFCL's native Plain implementation does not guarantee an identical local
model initialisation on every spawned client.  The first federation round
therefore uses an all-zero public reference: its "delta" is simply the full
locally-trained model, exactly matching native FATE's first full-state
aggregation.  After that aggregation every client has the same returned
global model, which becomes the reference for ordinary update-level DP.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from privacy.protection import gaussian_dp_upload, l2_norm


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _clone_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _as_single(payload: Any) -> Any:
    while isinstance(payload, (list, tuple)):
        if not payload:
            raise RuntimeError("FATE returned an empty UFCL native-DP payload.")
        payload = payload[0]
    return payload


def _server_values(ctx: Any, tag: str) -> list[Any]:
    guest_value = ctx.guest.get(tag)
    host_values = ctx.hosts.get(tag)
    if not isinstance(host_values, list):
        host_values = [host_values]
    return [guest_value] + host_values


def _broadcast(ctx: Any, tag: str, value: Any, host_count: int) -> None:
    ctx.guest.put(tag, value)
    ctx.hosts.put(tag, [value] * host_count if host_count > 1 else value)


def install_ufcl_native_delta_dp(ctx: Any, args: Any) -> None:
    """Install a process-local wrapper around FATE's native aggregation API."""
    if getattr(args, "_ufcl_native_delta_dp_installed", False):
        return
    if str(getattr(args, "protection", "plain")).lower() != "dp":
        return

    # Import only when an actual FATE process launches.  Keeping imports here
    # lets this project be syntax-checked on machines without the server's
    # bundled FATE package.
    from fate.ml.aggregator.aggregator_wrapper import (
        AggregatorClientWrapper,
        AggregatorServerWrapper,
    )
    if getattr(AggregatorClientWrapper, "_ufcl_native_delta_dp_patched", False):
        args._ufcl_native_delta_dp_installed = True
        return

    original_client_aggregation = AggregatorClientWrapper.model_aggregation
    original_server_aggregation = AggregatorServerWrapper.model_aggregation

    def client_model_aggregation(self: Any, aggregation_ctx: Any, model: torch.nn.Module):
        round_index = int(getattr(self, "_ufcl_native_dp_round", 0))
        parameters = _trainable_parameters(model)
        if not parameters:
            raise RuntimeError("UFCL native-DP found no trainable model parameters.")

        reference = getattr(self, "_ufcl_native_dp_reference", None)
        if reference is None:
            # Native Plain FATE aggregates full local model states in round 1.
            # A zero reference makes delta aggregation exactly that operation,
            # even when spawned clients have independently initialised models.
            reference = [torch.zeros_like(parameter.detach()) for parameter in parameters]
            self._ufcl_native_dp_reference = reference
            print(
                f"[UFCL-NativeDP] rank={aggregation_ctx.rank} round=1 uses zero reference "
                "to reproduce native full-state aggregation.",
                flush=True,
            )

        delta = [
            parameter.detach().clone().sub_(ref.to(device=parameter.device, dtype=parameter.dtype))
            for parameter, ref in zip(parameters, reference)
        ]

        configured_clip_norm = float(getattr(args, "dp_clip_norm", 0.0))
        clip_norm = getattr(self, "_ufcl_native_dp_clip_norm", None)
        if clip_norm is None and configured_clip_norm > 0.0:
            # A user-specified C is authoritative.  In particular, the
            # sigma=0 native-equivalence audit supplies an inactive bound so
            # that it can test the aggregation algebra without clipping.
            clip_norm = configured_clip_norm
            self._ufcl_native_dp_clip_norm = clip_norm
        if clip_norm is None:
            norm_tag = f"ufcl_native_dp_delta_norm_{round_index}"
            aggregation_ctx.arbiter.put(norm_tag, float(l2_norm(delta).item()))
            clip_norm = float(_as_single(aggregation_ctx.arbiter.get(norm_tag + "_clip")))
            self._ufcl_native_dp_clip_norm = clip_norm
            print(
                f"[DPCalibration] UFCL native rank={aggregation_ctx.rank} "
                f"epoch={round_index + 1} clip_norm={clip_norm:.8f}",
                flush=True,
            )

        seed = int(getattr(args, "dp_noise_seed", -1))
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + int(getattr(aggregation_ctx, "rank", 0)) * 1_000_003 + round_index)
        protected_delta, info = gaussian_dp_upload(
            delta,
            clip_norm=float(clip_norm),
            sigma=float(args.dp_sigma),
            post_clip_ratio=float(getattr(args, "dp_post_clip_ratio", 0.0)),
            generator=generator,
            experimental_noise_clip_ratio=float(getattr(args, "experimental_noise_clip_ratio", 0.0)),
            experimental_skip_update_clip=bool(getattr(args, "experimental_skip_update_clip", False)),
        )
        if round_index < 3 or (round_index + 1) % 50 == 0:
            clipped_signal = float(info["upload_l2_norm"]) * float(info["clip_coefficient"])
            ratio = float(info["noise_l2_norm"]) / max(clipped_signal, 1e-12)
            print(
                f"[UFCL-NativeDP] rank={aggregation_ctx.rank} round={round_index + 1} "
                f"raw_delta_l2={info['upload_l2_norm']:.8f} "
                f"clip={info['clip_coefficient']:.8f} noise_l2={info['noise_l2_norm']:.8f} "
                f"noise_to_signal={ratio:.8f}",
                flush=True,
            )

        # FATE serialises ``model.parameters()``.  Temporarily exposing the
        # protected delta therefore lets its original secure aggregation and
        # weighted averaging remain unchanged.
        local_after_train = _clone_parameters(parameters)
        with torch.no_grad():
            for parameter, protected in zip(parameters, protected_delta):
                parameter.copy_(protected.to(device=parameter.device, dtype=parameter.dtype))
        try:
            original_client_aggregation(self, aggregation_ctx, model)
            with torch.no_grad():
                for parameter, ref in zip(parameters, reference):
                    parameter.add_(ref.to(device=parameter.device, dtype=parameter.dtype))
        except Exception:
            # Avoid leaving a local model represented as a delta if FATE
            # aborts a federation round.  Restore the actual post-training
            # local model, not ``reference + delta``: the original FATE call
            # may have failed before it could write a received aggregate.
            with torch.no_grad():
                for parameter, local_value in zip(parameters, local_after_train):
                    parameter.copy_(local_value.to(device=parameter.device, dtype=parameter.dtype))
            raise

        self._ufcl_native_dp_reference = _clone_parameters(parameters)
        self._ufcl_native_dp_round = round_index + 1

    def server_model_aggregation(self: Any, aggregation_ctx: Any):
        round_index = int(getattr(self, "_ufcl_native_dp_round", 0))
        configured_clip_norm = float(getattr(args, "dp_clip_norm", 0.0))
        clip_norm = getattr(self, "_ufcl_native_dp_clip_norm", None)
        if clip_norm is None and configured_clip_norm > 0.0:
            clip_norm = configured_clip_norm
            self._ufcl_native_dp_clip_norm = clip_norm
        if clip_norm is None:
            norm_tag = f"ufcl_native_dp_delta_norm_{round_index}"
            norms = [float(value) for value in _server_values(aggregation_ctx, norm_tag)]
            clip_norm = float(np.quantile(norms, 0.9))
            _broadcast(aggregation_ctx, norm_tag + "_clip", clip_norm, len(norms) - 1)
            self._ufcl_native_dp_clip_norm = clip_norm
            print(
                f"[DPCalibration] UFCL native arbiter epoch={round_index + 1} "
                f"clip_norm={clip_norm:.8f}",
                flush=True,
            )

        original_server_aggregation(self, aggregation_ctx)
        self._ufcl_native_dp_round = round_index + 1

    AggregatorClientWrapper.model_aggregation = client_model_aggregation
    AggregatorServerWrapper.model_aggregation = server_model_aggregation
    AggregatorClientWrapper._ufcl_native_delta_dp_patched = True
    args._ufcl_native_delta_dp_installed = True
    print(
        f"[UFCL-NativeDP] rank={getattr(ctx, 'rank', '?')} installed native FATE delta-DP aggregation adapter.",
        flush=True,
    )
