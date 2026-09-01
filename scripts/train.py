import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vaak.audio.pipeline import AudioPipeline, AudioPipelineConfig
from vaak.config.settings import load_config
from vaak.core.logging import configure_logging, get_logger
from vaak.data import EvaluationDataset, TrainingDataset
from vaak.data.dataset import vaak_collate_fn
from vaak.data.manifest import load_manifest
from vaak.evaluation.evaluator import Evaluator
from vaak.models.factory import build_model_from_config
from vaak.training.trainer import Trainer
from vaak.utils.tools import get_optimal_device, set_seed
from vaak.utils.tracker import MLflowTracker
from vaak.utils.visualization import save_attack_benchmark_plot

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Vaak ASVspoof model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full training and evaluation pipeline with MLflow and registry."""
    set_seed(seed=42)
    configure_logging()
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    device = get_optimal_device()

    tracker = MLflowTracker(
        experiment_name=config.experiment_name,
        tracking_uri=project_root / "mlruns",
    )

    run_id = tracker.start_run(run_name=f"{config.model.name}_run")

    tracker.log_params(
        {
            "experiment_name": config.experiment_name,
            "model_name": config.model.name,
            "pretrained_model_name": config.model.pretrained_model_name,
            "layer_strategy": config.model.layer_strategy,
            "pooling_strategy": config.model.pooling_strategy,
            "sample_rate": config.data.sample_rate,
            "chunk_duration_seconds": config.data.chunk_duration_seconds,
            "hop_duration_seconds": config.data.hop_duration_seconds,
            "max_samples": config.data.max_samples,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "epochs": config.training.epochs,
            "device": device.type,
        }
    )

    logger.info(f"Starting experiment: {config.experiment_name} | Run ID: {run_id}")
    logger.info(f"Compute device: {device}")

    manifest_path = project_root / config.data.manifest
    manifest = load_manifest(manifest_path)

    audio_pipeline = AudioPipeline(
        config=AudioPipelineConfig(
            sample_rate=config.data.sample_rate,
            chunk_duration_seconds=config.data.chunk_duration_seconds,
            hop_duration_seconds=config.data.hop_duration_seconds,
        )
    )

    train_dataset = TrainingDataset(
        manifest=manifest,
        split="train",
        audio_pipeline=audio_pipeline,
        project_root=project_root,
        max_samples=config.data.max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=vaak_collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    val_dataset = EvaluationDataset(
        manifest=manifest,
        split="val",
        audio_pipeline=audio_pipeline,
        project_root=project_root,
        max_samples=2048,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=vaak_collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_dataset = EvaluationDataset(
        manifest=manifest,
        split="test",
        audio_pipeline=audio_pipeline,
        project_root=project_root,
        max_samples=2048,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=vaak_collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_config(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = project_root / "checkpoints" / config.experiment_name / run_id

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=checkpoint_dir,
        tracker=tracker,
    )

    try:
        logger.info("Beginning model training...")
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.training.epochs,
        )

        logger.info("Loading best checkpoint for final evaluation...")
        best_checkpoint_path = checkpoint_dir / "best_model.pt"
        if best_checkpoint_path.exists():
            checkpoint = torch.load(
                best_checkpoint_path, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])

        logger.info("Running full utterance-level evaluation on test split...")
        evaluator = Evaluator(model=model, device=device)
        report = evaluator.evaluate(test_loader)

        test_metrics = {
            "test_eer": report.eer,
            "test_norm_min_dcf": report.norm_min_dcf,
            "test_auc": report.auc,
        }

        # Log pairwise attack metrics
        for attack_id, metrics in report.attack_metrics.items():
            test_metrics[f"test_eer_{attack_id}"] = metrics["attack_vs_bonafide_eer"]
            test_metrics[f"test_norm_min_dcf_{attack_id}"] = metrics[
                "attack_vs_bonafide_norm_min_dcf"
            ]
            test_metrics[f"test_auc_{attack_id}"] = metrics["attack_vs_bonafide_auc"]

        tracker.log_metrics(test_metrics)

        logger.info(
            f"Final Test Metrics - EER: {report.eer:.4f} | "
            f"norm_minDCF: {report.norm_min_dcf:.4f} | "
            f"AUC: {report.auc:.4f}"
        )

        # Plot artifact
        plot_path = checkpoint_dir / "attack_benchmark_breakdown.png"
        save_attack_benchmark_plot(
            {
                atk: {
                    "eer": m["attack_vs_bonafide_eer"],
                    "auc": m["attack_vs_bonafide_auc"],
                    "min_dcf": m["attack_vs_bonafide_norm_min_dcf"],
                }
                for atk, m in report.attack_metrics.items()
            },
            plot_path,
        )
        tracker.log_artifact(plot_path)

        # Champion Registry Gate
        registry_dir = project_root / "registry"
        registry_dir.mkdir(exist_ok=True)
        champion_metrics_path = registry_dir / "champion_metrics.json"
        champion_model_path = registry_dir / "champion_model.pt"

        promote = False
        if not math.isinf(report.eer) and not math.isnan(report.eer):
            if not champion_metrics_path.exists():
                promote = True
            else:
                with open(champion_metrics_path) as f:
                    champion_metrics = json.load(f)

                if report.eer < champion_metrics.get("test_eer", float("inf")):
                    promote = True

        if promote:
            logger.info("🏆 New Champion Model! Promoting to registry...")
            if best_checkpoint_path.exists():
                shutil.copy2(best_checkpoint_path, champion_model_path)
                with open(champion_metrics_path, "w") as f:
                    json.dump(
                        {
                            "run_id": run_id,
                            "test_eer": report.eer,
                            "test_norm_min_dcf": report.norm_min_dcf,
                            "test_auc": report.auc,
                            "experiment_name": config.experiment_name,
                        },
                        f,
                        indent=4,
                    )
        else:
            logger.info(
                "Model did not beat current champion or produced invalid metrics."
            )

    finally:
        tracker.end_run()


if __name__ == "__main__":
    main()
