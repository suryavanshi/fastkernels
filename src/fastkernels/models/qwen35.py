"""Shape metadata for Qwen3.5 MoE models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Qwen35MoESpec:
    """Kernel-facing shape spec for Qwen3.5-style MoE models."""

    name: str
    hidden_size: int
    num_layers: int
    num_experts: int
    num_routed_experts: int
    num_shared_experts: int
    expert_intermediate_size: int
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    rope_theta: float | None = None
    vocab_size: int | None = None

    @property
    def active_experts_per_token(self) -> int:
        return self.num_routed_experts + self.num_shared_experts

    @property
    def gated_up_features(self) -> int:
        return 2 * self.expert_intermediate_size

    def layer_kinds(self) -> tuple[str, ...]:
        """Return the 40-layer Qwen3.5 hybrid pattern."""

        pattern = ("deltanet_moe", "deltanet_moe", "deltanet_moe", "attention_moe")
        repeats, remainder = divmod(self.num_layers, len(pattern))
        return pattern * repeats + pattern[:remainder]

    def moe_shapes(self, tokens: int) -> dict[str, tuple[int, ...]]:
        """Return common MoE tensor shapes for a token count."""

        routed_tokens = tokens * self.num_routed_experts
        return {
            "hidden_states": (tokens, self.hidden_size),
            "router_logits": (tokens, self.num_experts),
            "topk_ids": (tokens, self.num_routed_experts),
            "topk_weights": (tokens, self.num_routed_experts),
            "routed_hidden_states": (routed_tokens, self.hidden_size),
            "expert_gate_up_weight": (
                self.num_experts,
                self.gated_up_features,
                self.hidden_size,
            ),
            "expert_down_weight": (
                self.num_experts,
                self.hidden_size,
                self.expert_intermediate_size,
            ),
            "expert_gate_up": (routed_tokens, self.gated_up_features),
            "expert_activation": (routed_tokens, self.expert_intermediate_size),
            "expert_output": (routed_tokens, self.hidden_size),
        }

    def summary_lines(self, token_counts: Iterable[int] = (1, 16, 128)) -> list[str]:
        lines = [
            f"model: {self.name}",
            f"layers: {self.num_layers}",
            f"hidden_size: {self.hidden_size}",
            f"experts: {self.num_experts}",
            f"routed/shared active experts: {self.num_routed_experts}/{self.num_shared_experts}",
            f"expert_intermediate_size: {self.expert_intermediate_size}",
            f"layer pattern: {', '.join(self.layer_kinds()[:4])} x {self.num_layers // 4}",
        ]
        for tokens in token_counts:
            routed = tokens * self.num_routed_experts
            lines.append(f"tokens={tokens}: routed_expert_rows={routed}")
        return lines

    @classmethod
    def from_hf_config(cls, config: dict[str, Any], name: str = "hf-config") -> "Qwen35MoESpec":
        """Build a spec from a Hugging Face config dictionary.

        Qwen-family config fields have changed across releases, so this accepts
        the common aliases used by recent MoE configs.
        """

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
            num_attention_heads=first("num_attention_heads"),
            num_key_value_heads=first("num_key_value_heads"),
            rope_theta=first("rope_theta"),
            vocab_size=first("vocab_size"),
        )

    @classmethod
    def from_json_file(cls, path: str | Path, name: str | None = None) -> "Qwen35MoESpec":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls.from_hf_config(config, name=name or config_path.stem)


def qwen35_35b_a3b_spec() -> Qwen35MoESpec:
    """Return the known Qwen3.5-35B-A3B kernel-facing shape spec."""

    return Qwen35MoESpec(
        name="Qwen3.5-35B-A3B",
        hidden_size=2048,
        num_layers=40,
        num_experts=256,
        num_routed_experts=8,
        num_shared_experts=1,
        expert_intermediate_size=512,
    )
