from pathlib import Path

from vaak.utils.visualization import save_attack_benchmark_plot


def test_save_attack_benchmark_plot(tmp_path: Path) -> None:
    dummy_metrics = {
        "A07": {"eer": 0.04, "auc": 0.98, "min_dcf": 0.12},
        "A08": {"eer": 0.18, "auc": 0.65, "min_dcf": 0.45},
        "A09": {"eer": 0.12, "auc": 0.78, "min_dcf": 0.30},
    }
    output_image = tmp_path / "test_attack_plot.png"
    save_attack_benchmark_plot(dummy_metrics, output_image)

    assert output_image.exists()
    assert output_image.stat().st_size > 0
