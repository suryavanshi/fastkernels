"""Shape metadata for Qwen3.6 hybrid DeltaNet/Attention MoE models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Qwen36A3BSpec:
    """Kernel-facing shape spec for Qwen3.6-35B-A3B-style models."""

    name: str
    hidden_size: int
    num_layers: int
    num_experts: int
    num_routed_experts: int
    num_shared_experts: int
    expert_intermediate_size: int
    deltanet_value_heads: int
    deltanet_qk_heads: int
    deltanet_head_dim: int
    attention_heads: int
    attention_kv_heads: int
    attention_head_dim: int
    rope_dim: int
    vocab_size: int
    context_length: int

    @property
    def active_experts_per_token(self) -> int:
        return self.num_routed_experts + self.num_shared_experts

    @property
    def gated_up_features(self) -> int:
        return 2 * self.expert_intermediate_size

    @property
    def deltanet_value_dim_per_qk_head(self) -> int:
        if self.deltanet_value_heads % self.deltanet_qk_heads:
            raise ValueError("deltanet_value_heads must be divisible by deltanet_qk_heads")
        return (self.deltanet_value_heads // self.deltanet_qk_heads) * self.deltanet_head_dim

    @property
    def attention_heads_per_kv_head(self) -> int:
        if self.attention_heads % self.attention_kv_heads:
            raise ValueError("attention_heads must be divisible by attention_kv_heads")
        return self.attention_heads // self.attention_kv_heads

    def layer_kinds(self) -> tuple[str, ...]:
        pattern = ("deltanet_moe", "deltanet_moe", "deltanet_moe", "attention_moe")
        repeats, remainder = divmod(self.num_layers, len(pattern))
        return pattern * repeats + pattern[:remainder]

    def layer_counts(self) -> dict[str, int]:
        kinds = self.layer_kinds()
        return {
            "deltanet_moe": kinds.count("deltanet_moe"),
            "attention_moe": kinds.count("attention_moe"),
        }

    def deltanet_state_shape(self) -> tuple[int, int, int]:
        return (
            self.deltanet_qk_heads,
            self.deltanet_head_dim,
            self.deltanet_value_dim_per_qk_head,
        )

    def attention_cache_shape(self, max_positions: int | None = None) -> tuple[int, int, int]:
        return (
            max_positions or self.context_length,
            self.attention_kv_heads,
            self.attention_head_dim,
        )

    def decode_shapes(self, max_positions: int | None = None) -> dict[str, tuple[int, ...]]:
        counts = self.layer_counts()
        return {
            "hidden_state": (self.hidden_size,),
            "deltanet_states": (counts["deltanet_moe"], *self.deltanet_state_shape()),
            "attention_key_cache": (counts["attention_moe"], *self.attention_cache_shape(max_positions)),
            "attention_value_cache": (counts["attention_moe"], *self.attention_cache_shape(max_positions)),
            "router_logits": (self.num_experts,),
            "expert_gate_up_weight": (self.num_experts, self.gated_up_features, self.hidden_size),
            "expert_down_weight": (self.num_experts, self.hidden_size, self.expert_intermediate_size),
        }

    def summary_lines(self, token_counts: Iterable[int] = (1, 16, 128)) -> list[str]:
        counts = self.layer_counts()
        lines = [
            f"model: {self.name}",
            f"layers: {self.num_layers}",
            f"hidden_size: {self.hidden_size}",
            f"layer pattern: {', '.join(self.layer_kinds()[:4])} x {self.num_layers // 4}",
            f"deltanet/attention layers: {counts['deltanet_moe']}/{counts['attention_moe']}",
            f"deltanet qk/value heads: {self.deltanet_qk_heads}/{self.deltanet_value_heads}",
            f"attention q/kv heads: {self.attention_heads}/{self.attention_kv_heads}",
            f"experts: {self.num_experts}",
            f"routed/shared active experts: {self.num_routed_experts}/{self.num_shared_experts}",
            f"expert_intermediate_size: {self.expert_intermediate_size}",
        ]
        for tokens in token_counts:
            routed = tokens * self.num_routed_experts
            lines.append(f"tokens={tokens}: routed_expert_rows={routed}")
        return lines

    @classmethod
    def from_hf_config(cls, config: dict[str, Any], name: str = "hf-config") -> "Qwen36A3BSpec":
        def first(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in config and config[key] is not None:
                    return config[key]
            return default

        return cls(
            name=name,
            hidden_size=int(first("hidden_size")),
            num_layers=int(first("num_hidden_layers", "num_layers")),
            num_experts=int(first("num_experts", "n_routed_experts")),
            num_routed_experts=int(first("num_experts_per_tok", "num_experts_per_token", default=8)),
            num_shared_experts=int(first("n_shared_experts", "num_shared_experts", default=1)),
            expert_intermediate_size=int(
                first("moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
            ),
            deltanet_value_heads=int(first("num_linear_attention_value_heads", default=32)),
            deltanet_qk_heads=int(first("num_linear_attention_qk_heads", default=16)),
            deltanet_head_dim=int(first("linear_attention_head_dim", default=128)),
            attention_heads=int(first("num_attention_heads", default=16)),
            attention_kv_heads=int(first("num_key_value_heads", default=2)),
            attention_head_dim=int(first("attention_head_dim", "head_dim", default=256)),
            rope_dim=int(first("rope_dim", "rotary_emb_base_dim", default=64)),
            vocab_size=int(first("vocab_size", default=248320)),
            context_length=int(first("max_position_embeddings", "seq_length", default=262144)),
        )

    @classmethod
    def from_json_file(cls, path: str | Path, name: str | None = None) -> "Qwen36A3BSpec":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls.from_hf_config(config, name=name or config_path.stem)


def qwen36_35b_a3b_spec() -> Qwen36A3BSpec:
    """Return the public Qwen3.6-35B-A3B kernel-facing shape spec."""

    return Qwen36A3BSpec(
        name="Qwen3.6-35B-A3B",
        hidden_size=2048,
        num_layers=40,
        num_experts=256,
        num_routed_experts=8,
        num_shared_experts=1,
        expert_intermediate_size=512,
        deltanet_value_heads=32,
        deltanet_qk_heads=16,
        deltanet_head_dim=128,
        attention_heads=16,
        attention_kv_heads=2,
        attention_head_dim=256,
        rope_dim=64,
        vocab_size=248320,
        context_length=262144,
    )


def synthetic_qwen36_spec() -> Qwen36A3BSpec:
    """Return a tiny Qwen3.6-shaped spec for correctness tests."""

    return Qwen36A3BSpec(
        name="synthetic-qwen3.6-a3b",
        hidden_size=16,
        num_layers=4,
        num_experts=4,
        num_routed_experts=2,
        num_shared_experts=1,
        expert_intermediate_size=8,
        deltanet_value_heads=4,
        deltanet_qk_heads=2,
        deltanet_head_dim=4,
        attention_heads=4,
        attention_kv_heads=2,
        attention_head_dim=4,
        rope_dim=4,
        vocab_size=64,
        context_length=16,
    )
