"""Declarative PI0.5 true-full capture contract and buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from vla_lens.traces import ModelSiteSpec

DEFAULT_PI05_VLM_LAYERS = tuple(range(18))
DEFAULT_PI05_EXPERT_LAYERS = tuple(range(18))

PI05_FULL_CAPTURE_IMPLEMENTED = True

PREFIX_TOKEN_SPACE = "pi05.prefix"
ACTION_TOKEN_SPACE = "pi05.action_suffix"
EXPERT_CONTEXT_TOKEN_SPACE = "pi05.expert_context"


class IncompletePI05FullCaptureError(RuntimeError):
    """Raised when true-full capture materialization is requested before completion."""


@dataclass(frozen=True, slots=True)
class SiteCaptureDeclaration:
    """Static contract for one raw tensor site in true-full capture."""

    name: str
    axes: tuple[str, ...]
    module: str
    tensor_type: str
    family: str
    role: str
    segment: str
    layer: int | None = None
    token_kind: str | None = None
    token_space_id: str | None = None
    query_token_space_id: str | None = None
    key_token_space_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def spec(self, array: Any, metadata: Mapping[str, Any] | None = None) -> ModelSiteSpec:
        """Create a raw/exact ``ModelSiteSpec`` for a captured value."""

        merged_metadata = {
            "site_family": self.family,
            "site_role": self.role,
            "site_segment": self.segment,
            "site_axes": self.axes,
            "capture_profile": "audit_full",
            "required_for_true_full": True,
            **dict(self.metadata),
        }
        if metadata:
            merged_metadata.update(metadata)
        return ModelSiteSpec(
            name=self.name,
            array=_to_numpy(array),
            axes=self.axes,
            module=self.module,
            layer=self.layer,
            tensor_type=self.tensor_type,
            token_kind=self.token_kind,
            metadata=merged_metadata,
            family=self.family,
            role=self.role,
            segment=self.segment,
            materialization="raw",
            exactness="exact",
            token_space_id=self.token_space_id,
            query_token_space_id=self.query_token_space_id,
            key_token_space_id=self.key_token_space_id,
            capture_family=self.family,
            view_kind=_view_kind(self.family, self.role),
            capture_role="raw_debug",
            default_view=False,
        )


@dataclass(frozen=True, slots=True)
class CapturedSite:
    """Captured value plus optional per-capture metadata for one declared site."""

    declaration: SiteCaptureDeclaration
    array: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def spec(self) -> ModelSiteSpec:
        return self.declaration.spec(self.array, self.metadata)


class PerSiteCaptureBuffer:
    """Small in-memory buffer keyed by declared model site name."""

    def __init__(self, declarations: Iterable[SiteCaptureDeclaration]):
        self._declarations = {declaration.name: declaration for declaration in declarations}
        self._captures: dict[str, CapturedSite] = {}

    @property
    def declarations(self) -> tuple[SiteCaptureDeclaration, ...]:
        return tuple(self._declarations.values())

    @property
    def captured_names(self) -> tuple[str, ...]:
        return tuple(self._captures)

    def capture(
        self,
        name: str,
        array: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one raw tensor for a declared site."""

        try:
            declaration = self._declarations[name]
        except KeyError as exc:
            raise KeyError(f"Unknown PI0.5 full capture site: {name}") from exc
        self._captures[name] = CapturedSite(
            declaration=declaration,
            array=_to_numpy(array),
            metadata={} if metadata is None else dict(metadata),
        )

    def missing_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._declarations if name not in self._captures)

    def is_complete(self) -> bool:
        return not self.missing_names()

    def specs(self, *, require_complete: bool = True) -> tuple[ModelSiteSpec, ...]:
        """Materialize captured sites as raw/exact ``ModelSiteSpec`` records."""

        missing = self.missing_names()
        if require_complete and missing:
            raise IncompletePI05FullCaptureError(
                f"PI0.5 full capture is incomplete; missing {len(missing)} required sites"
            )
        return tuple(capture.spec() for capture in self._captures.values())


@dataclass(frozen=True, slots=True)
class PI05FullCaptureStatus:
    """Readiness report for the true-full capture contract."""

    implemented: bool
    complete: bool
    required_count: int
    captured_count: int
    missing_sites: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.implemented and self.complete


def pi05_full_capture_status(
    captured_names: Iterable[str],
    *,
    declarations: Iterable[SiteCaptureDeclaration] | None = None,
) -> PI05FullCaptureStatus:
    """Report whether true-full capture can be enabled for the captured sites."""

    declared = tuple(pi05_full_site_declarations() if declarations is None else declarations)
    captured = set(captured_names)
    missing = tuple(
        declaration.name for declaration in declared if declaration.name not in captured
    )
    return PI05FullCaptureStatus(
        implemented=PI05_FULL_CAPTURE_IMPLEMENTED,
        complete=not missing,
        required_count=len(declared),
        captured_count=len(captured),
        missing_sites=missing,
    )


def required_pi05_full_site_names(
    *,
    vlm_layers: Iterable[int] = DEFAULT_PI05_VLM_LAYERS,
    expert_layers: Iterable[int] = DEFAULT_PI05_EXPERT_LAYERS,
) -> tuple[str, ...]:
    """Return the required raw site names for true-full PI0.5 capture."""

    return tuple(
        declaration.name
        for declaration in pi05_full_site_declarations(
            vlm_layers=vlm_layers,
            expert_layers=expert_layers,
        )
    )


def missing_pi05_full_sites(
    captured_names: Iterable[str],
    *,
    vlm_layers: Iterable[int] = DEFAULT_PI05_VLM_LAYERS,
    expert_layers: Iterable[int] = DEFAULT_PI05_EXPERT_LAYERS,
) -> tuple[str, ...]:
    """Return required true-full sites not present in ``captured_names``."""

    captured = set(captured_names)
    return tuple(
        name
        for name in required_pi05_full_site_names(
            vlm_layers=vlm_layers,
            expert_layers=expert_layers,
        )
        if name not in captured
    )


def make_raw_model_site_spec(
    declaration: SiteCaptureDeclaration,
    array: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ModelSiteSpec:
    """Create a raw/exact ``ModelSiteSpec`` from one declaration."""

    return declaration.spec(array, metadata)


def materialize_raw_model_site_specs(
    captures: Mapping[str, Any],
    *,
    declarations: Iterable[SiteCaptureDeclaration] | None = None,
    require_complete: bool = True,
) -> tuple[ModelSiteSpec, ...]:
    """Materialize captured arrays according to the true-full declaration catalog."""

    buffer = PerSiteCaptureBuffer(
        pi05_full_site_declarations() if declarations is None else declarations
    )
    for name, array in captures.items():
        buffer.capture(name, array)
    return buffer.specs(require_complete=require_complete)


def pi05_full_site_declarations(
    *,
    vlm_layers: Iterable[int] = DEFAULT_PI05_VLM_LAYERS,
    expert_layers: Iterable[int] = DEFAULT_PI05_EXPERT_LAYERS,
) -> tuple[SiteCaptureDeclaration, ...]:
    """Build required raw PI0.5 full-capture site declarations."""

    declarations: list[SiteCaptureDeclaration] = []
    declarations.extend(_input_declarations())
    for layer in vlm_layers:
        declarations.extend(_transformer_layer_declarations("vlm", int(layer), by_step=False))
    for layer in expert_layers:
        declarations.extend(_transformer_layer_declarations("expert", int(layer), by_step=True))
    declarations.extend(_action_head_declarations())
    return tuple(declarations)


def _input_declarations() -> tuple[SiteCaptureDeclaration, ...]:
    return (
        _declaration(
            name="pi05.vlm.prefix.input_embeddings",
            axes=("policy_call", "token", "channel"),
            module="pi05.vlm.prefix",
            tensor_type="embedding",
            family="embedding",
            role="input_embeddings",
            segment="vlm_prefix",
            token_kind="prefix",
            token_space_id=PREFIX_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.expert.by_step.input_embeddings",
            axes=("policy_call", "generation_step", "token", "channel"),
            module="pi05.expert",
            tensor_type="embedding",
            family="embedding",
            role="input_embeddings",
            segment="action_expert",
            token_kind="action",
            token_space_id=ACTION_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.inputs.attention_mask",
            axes=("policy_call", "token"),
            module="pi05.inputs",
            tensor_type="mask",
            family="mask",
            role="attention_mask",
            segment="inputs",
            token_kind="mixed",
            token_space_id=PREFIX_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.inputs.causal_mask",
            axes=("policy_call", "query_token", "key_token"),
            module="pi05.inputs",
            tensor_type="mask",
            family="mask",
            role="causal_mask",
            segment="inputs",
            query_token_space_id=PREFIX_TOKEN_SPACE,
            key_token_space_id=PREFIX_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.expert.by_step.attention_mask",
            axes=("policy_call", "generation_step", "token"),
            module="pi05.expert.inputs",
            tensor_type="mask",
            family="mask",
            role="attention_mask",
            segment="action_expert",
            token_kind="action",
            token_space_id=ACTION_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.expert.by_step.causal_mask",
            axes=("policy_call", "generation_step", "query_token", "key_token"),
            module="pi05.expert.inputs",
            tensor_type="mask",
            family="mask",
            role="causal_mask",
            segment="action_expert",
            query_token_space_id=ACTION_TOKEN_SPACE,
            key_token_space_id=ACTION_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.inputs.position_ids",
            axes=("policy_call", "token"),
            module="pi05.inputs",
            tensor_type="position",
            family="position",
            role="position_ids",
            segment="inputs",
            token_kind="mixed",
            token_space_id=PREFIX_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.expert.by_step.position_ids",
            axes=("policy_call", "generation_step", "token"),
            module="pi05.expert.inputs",
            tensor_type="position",
            family="position",
            role="position_ids",
            segment="action_expert",
            token_kind="action",
            token_space_id=ACTION_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.inputs.rope.cos",
            axes=("policy_call", "token", "head_channel"),
            module="pi05.inputs.rope",
            tensor_type="rope",
            family="position",
            role="rope_cos",
            segment="rope",
            token_kind="mixed",
            token_space_id=PREFIX_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.inputs.rope.sin",
            axes=("policy_call", "token", "head_channel"),
            module="pi05.inputs.rope",
            tensor_type="rope",
            family="position",
            role="rope_sin",
            segment="rope",
            token_kind="mixed",
            token_space_id=PREFIX_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.expert.by_step.rope.cos",
            axes=("policy_call", "generation_step", "token", "head_channel"),
            module="pi05.expert.rope",
            tensor_type="rope",
            family="position",
            role="rope_cos",
            segment="action_expert",
            token_kind="action",
            token_space_id=ACTION_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.expert.by_step.rope.sin",
            axes=("policy_call", "generation_step", "token", "head_channel"),
            module="pi05.expert.rope",
            tensor_type="rope",
            family="position",
            role="rope_sin",
            segment="action_expert",
            token_kind="action",
            token_space_id=ACTION_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.inputs.rope.metadata",
            axes=("policy_call", "rope_field"),
            module="pi05.inputs.rope",
            tensor_type="rope",
            family="position",
            role="rope_metadata",
            segment="rope",
            token_kind="mixed",
            token_space_id=PREFIX_TOKEN_SPACE,
        ),
    )


def _transformer_layer_declarations(
    stack: str,
    layer: int,
    *,
    by_step: bool,
) -> tuple[SiteCaptureDeclaration, ...]:
    segment = "action_expert" if stack == "expert" else "vlm_prefix"
    token_kind = "action" if stack == "expert" else "prefix"
    token_space = ACTION_TOKEN_SPACE if stack == "expert" else PREFIX_TOKEN_SPACE
    key_token_space = EXPERT_CONTEXT_TOKEN_SPACE if stack == "expert" else PREFIX_TOKEN_SPACE
    hidden_axes = _hidden_axes(by_step)
    qkv_axes = _qkv_axes(by_step)
    attention_axes = _attention_axes(by_step)
    module = f"pi05.{stack}.layers.{layer}"

    declarations = [
        _declaration(
            name=f"{module}.residual_pre_attention",
            axes=hidden_axes,
            module=module,
            layer=layer,
            tensor_type="residual",
            family="residual",
            role="residual_pre_attention",
            segment=segment,
            token_kind=token_kind,
            token_space_id=token_space,
        ),
        _declaration(
            name=f"{module}.attention_norm_output",
            axes=hidden_axes,
            module=module,
            layer=layer,
            tensor_type="norm",
            family="normalization",
            role="attention_norm_output",
            segment=segment,
            token_kind=token_kind,
            token_space_id=token_space,
        ),
    ]
    if stack == "expert":
        declarations.extend(
            _adarms_declarations(
                module,
                layer,
                hidden_axes,
                token_kind,
                token_space,
                norm_site="attention_adarms",
            )
        )

    declarations.extend(
        [
            _declaration(
                name=f"{module}.attention.q",
                axes=qkv_axes,
                module=f"{module}.attention",
                layer=layer,
                tensor_type="attention_q",
                family="attention",
                role="q",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.attention.k",
                axes=qkv_axes,
                module=f"{module}.attention",
                layer=layer,
                tensor_type="attention_k",
                family="attention",
                role="k",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.attention.v",
                axes=qkv_axes,
                module=f"{module}.attention",
                layer=layer,
                tensor_type="attention_v",
                family="attention",
                role="v",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.attention.pre_mask_scores",
                axes=attention_axes,
                module=f"{module}.attention",
                layer=layer,
                tensor_type="attention_scores",
                family="attention",
                role="pre_mask_scores",
                segment=segment,
                token_kind=token_kind,
                query_token_space_id=token_space,
                key_token_space_id=key_token_space,
            ),
            _declaration(
                name=f"{module}.attention.post_mask_logits",
                axes=attention_axes,
                module=f"{module}.attention",
                layer=layer,
                tensor_type="attention_logits",
                family="attention",
                role="post_mask_logits",
                segment=segment,
                token_kind=token_kind,
                query_token_space_id=token_space,
                key_token_space_id=key_token_space,
            ),
            _declaration(
                name=f"{module}.attention.attention_probs",
                axes=attention_axes,
                module=f"{module}.attention",
                layer=layer,
                tensor_type="attention_probs",
                family="attention",
                role="attention_probs",
                segment=segment,
                token_kind=token_kind,
                query_token_space_id=token_space,
                key_token_space_id=key_token_space,
            ),
            _declaration(
                name=f"{module}.attention.attn_output_pre_o_proj",
                axes=hidden_axes,
                module=f"{module}.attention",
                layer=layer,
                tensor_type="attention_output",
                family="attention",
                role="attn_output_pre_o_proj",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.attention.o_proj",
                axes=hidden_axes,
                module=f"{module}.attention.o_proj",
                layer=layer,
                tensor_type="attention_output",
                family="attention",
                role="o_proj",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.residual_post_attention",
                axes=hidden_axes,
                module=module,
                layer=layer,
                tensor_type="residual",
                family="residual",
                role="residual_post_attention",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.residual_pre_mlp",
                axes=hidden_axes,
                module=module,
                layer=layer,
                tensor_type="residual",
                family="residual",
                role="residual_pre_mlp",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.mlp_norm_output",
                axes=hidden_axes,
                module=module,
                layer=layer,
                tensor_type="norm",
                family="normalization",
                role="mlp_norm_output",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            *(
                _adarms_declarations(
                    module,
                    layer,
                    hidden_axes,
                    token_kind,
                    token_space,
                    norm_site="mlp_adarms",
                )
                if stack == "expert"
                else ()
            ),
            *_mlp_declarations(module, layer, hidden_axes, segment, token_kind, token_space),
            _declaration(
                name=f"{module}.residual_post_mlp",
                axes=hidden_axes,
                module=module,
                layer=layer,
                tensor_type="residual",
                family="residual",
                role="residual_post_mlp",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.kv_cache.key",
                axes=_kv_cache_axes(by_step),
                module=f"{module}.attention",
                layer=layer,
                tensor_type="kv_cache",
                family="cache",
                role="kv_cache_key",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
            _declaration(
                name=f"{module}.kv_cache.value",
                axes=_kv_cache_axes(by_step),
                module=f"{module}.attention",
                layer=layer,
                tensor_type="kv_cache",
                family="cache",
                role="kv_cache_value",
                segment=segment,
                token_kind=token_kind,
                token_space_id=token_space,
            ),
        ]
    )
    return tuple(declarations)


def _adarms_declarations(
    module: str,
    layer: int,
    axes: tuple[str, ...],
    token_kind: str,
    token_space: str,
    norm_site: str,
) -> tuple[SiteCaptureDeclaration, ...]:
    return (
        _declaration(
            name=f"{module}.{norm_site}.scale",
            axes=axes,
            module=f"{module}.{norm_site}",
            layer=layer,
            tensor_type="adarms",
            family="normalization",
            role="adarms_scale",
            segment="action_expert",
            token_kind=token_kind,
            token_space_id=token_space,
        ),
        _declaration(
            name=f"{module}.{norm_site}.shift",
            axes=axes,
            module=f"{module}.{norm_site}",
            layer=layer,
            tensor_type="adarms",
            family="normalization",
            role="adarms_shift",
            segment="action_expert",
            token_kind=token_kind,
            token_space_id=token_space,
        ),
        _declaration(
            name=f"{module}.{norm_site}.gate",
            axes=axes,
            module=f"{module}.{norm_site}",
            layer=layer,
            tensor_type="adarms",
            family="normalization",
            role="adarms_gate",
            segment="action_expert",
            token_kind=token_kind,
            token_space_id=token_space,
        ),
    )


def _mlp_declarations(
    module: str,
    layer: int,
    axes: tuple[str, ...],
    segment: str,
    token_kind: str,
    token_space: str,
) -> tuple[SiteCaptureDeclaration, ...]:
    return tuple(
        _declaration(
            name=f"{module}.mlp.{site}",
            axes=axes,
            module=f"{module}.mlp",
            layer=layer,
            tensor_type="mlp",
            family="mlp",
            role=role,
            segment=segment,
            token_kind=token_kind,
            token_space_id=token_space,
        )
        for site, role in (
            ("gate", "mlp_gate"),
            ("up", "mlp_up"),
            ("intermediate", "mlp_intermediate"),
            ("down", "mlp_down"),
            ("output", "mlp_output"),
        )
    )


def _action_head_declarations() -> tuple[SiteCaptureDeclaration, ...]:
    return (
        _declaration(
            name="pi05.action_head.input",
            axes=("policy_call", "generation_step", "token", "channel"),
            module="pi05.action_head",
            tensor_type="action_head",
            family="action_head",
            role="action_head_input",
            segment="action_head",
            token_kind="action",
            token_space_id=ACTION_TOKEN_SPACE,
        ),
        _declaration(
            name="pi05.action_head.output",
            axes=("policy_call", "generation_step", "horizon", "action_dim"),
            module="pi05.action_head",
            tensor_type="action_head",
            family="action_head",
            role="action_head_output",
            segment="action_head",
            token_kind="action",
            token_space_id=ACTION_TOKEN_SPACE,
        ),
    )


def _declaration(
    *,
    name: str,
    axes: tuple[str, ...],
    module: str,
    tensor_type: str,
    family: str,
    role: str,
    segment: str,
    layer: int | None = None,
    token_kind: str | None = None,
    token_space_id: str | None = None,
    query_token_space_id: str | None = None,
    key_token_space_id: str | None = None,
) -> SiteCaptureDeclaration:
    return SiteCaptureDeclaration(
        name=name,
        axes=axes,
        module=module,
        layer=layer,
        tensor_type=tensor_type,
        family=family,
        role=role,
        segment=segment,
        token_kind=token_kind,
        token_space_id=token_space_id,
        query_token_space_id=query_token_space_id,
        key_token_space_id=key_token_space_id,
    )


def _view_kind(family: str, role: str) -> str:
    if family == "attention":
        return "attention"
    if family == "cache":
        return "cache"
    if family == "action_head":
        return "action"
    if family in {"mlp", "normalization", "residual"}:
        return "internals"
    if role in {
        "attention_mask",
        "causal_mask",
        "position_ids",
        "rope_cos",
        "rope_sin",
        "rope_metadata",
    }:
        return "raw"
    return "features"


def _hidden_axes(by_step: bool) -> tuple[str, ...]:
    if by_step:
        return ("policy_call", "generation_step", "token", "channel")
    return ("policy_call", "token", "channel")


def _qkv_axes(by_step: bool) -> tuple[str, ...]:
    if by_step:
        return ("policy_call", "generation_step", "head", "token", "head_channel")
    return ("policy_call", "head", "token", "head_channel")


def _attention_axes(by_step: bool) -> tuple[str, ...]:
    if by_step:
        return ("policy_call", "generation_step", "head", "query_token", "key_token")
    return ("policy_call", "head", "query_token", "key_token")


def _kv_cache_axes(by_step: bool) -> tuple[str, ...]:
    if by_step:
        return ("policy_call", "generation_step", "kv_head", "cached_token", "head_channel")
    return ("policy_call", "kv_head", "cached_token", "head_channel")


def _to_numpy(array: Any) -> np.ndarray:
    detach = getattr(array, "detach", None)
    if callable(detach):
        array = detach()
    cpu = getattr(array, "cpu", None)
    if callable(cpu):
        array = cpu()
    return np.asarray(array)


__all__ = [
    "ACTION_TOKEN_SPACE",
    "DEFAULT_PI05_EXPERT_LAYERS",
    "DEFAULT_PI05_VLM_LAYERS",
    "EXPERT_CONTEXT_TOKEN_SPACE",
    "IncompletePI05FullCaptureError",
    "PI05_FULL_CAPTURE_IMPLEMENTED",
    "PI05FullCaptureStatus",
    "PREFIX_TOKEN_SPACE",
    "CapturedSite",
    "PerSiteCaptureBuffer",
    "SiteCaptureDeclaration",
    "make_raw_model_site_spec",
    "materialize_raw_model_site_specs",
    "missing_pi05_full_sites",
    "pi05_full_capture_status",
    "pi05_full_site_declarations",
    "required_pi05_full_site_names",
]
