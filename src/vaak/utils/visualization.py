import math
from pathlib import Path

import matplotlib.pyplot as plt


def save_attack_benchmark_plot(
    attack_metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Generate a dual-panel plot for per-attack EER and AUC metrics."""
    if not attack_metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            "No attack-specific metrics available.",
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        ax.axis("off")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    attacks = sorted(attack_metrics.keys())
    eer_values = [attack_metrics[atk]["eer"] for atk in attacks]
    auc_values = [attack_metrics[atk]["auc"] for atk in attacks]

    # Cap INF values to 1.0 just for setting bar heights and axis limits
    eer_plot_values = [1.0 if math.isinf(v) else v for v in eer_values]

    fig, (ax_eer, ax_auc) = plt.subplots(1, 2, figsize=(16, 6))

    # EER Plot (Lower is Better)
    # Green (<5%), Yellow (5-15%), Red (>15%)
    eer_colors = [
        "#2ecc71" if val <= 0.05 else "#f39c12" if val <= 0.15 else "#e74c3c"
        for val in eer_plot_values
    ]
    bars_eer = ax_eer.bar(attacks, eer_plot_values, color=eer_colors)
    ax_eer.axhline(
        0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.50)"
    )

    max_eer = max(eer_plot_values) if eer_plot_values else 0.5
    ax_eer.set_ylim(0.0, max(max_eer * 1.15, 0.55))
    ax_eer.set_xlabel("Attack System")
    ax_eer.set_ylabel("Equal Error Rate (EER)")
    ax_eer.set_title("Per-Attack EER (Lower is Better)")
    ax_eer.legend()
    ax_eer.tick_params(axis="x", rotation=45)

    # Use original eer_values to label the bars (printing "N/A" if infinite)
    for bar, orig_val in zip(bars_eer, eer_values, strict=True):
        height = bar.get_height()
        label_text = "N/A" if math.isinf(orig_val) else f"{orig_val:.3f}"
        ax_eer.annotate(
            label_text,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # AUC Plot (Higher is Better)
    # Green (>85%), Yellow (70-85%), Red (<70%)
    auc_colors = [
        "#2ecc71" if val >= 0.85 else "#f39c12" if val >= 0.70 else "#e74c3c"
        for val in auc_values
    ]
    bars_auc = ax_auc.bar(attacks, auc_values, color=auc_colors)
    ax_auc.axhline(
        0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.50)"
    )
    ax_auc.set_ylim(0.0, 1.05)
    ax_auc.set_xlabel("Attack System")
    ax_auc.set_ylabel("Area Under ROC Curve (AUC)")
    ax_auc.set_title("Per-Attack ROC-AUC (Higher is Better)")
    ax_auc.legend()
    ax_auc.tick_params(axis="x", rotation=45)

    for bar in bars_auc:
        height = bar.get_height()
        ax_auc.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.suptitle("ASVspoof Attack-Level Benchmark Diagnostics", fontsize=14, y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
