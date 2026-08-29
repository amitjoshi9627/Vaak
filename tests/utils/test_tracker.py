from pathlib import Path

from vaak.utils.tracker import MLflowTracker


def test_tracker_lifecycle(tmp_path: Path) -> None:
    db_dir = tmp_path / "experiments"
    tracker = MLflowTracker(
        experiment_name="test_experiment",
        tracking_uri=db_dir,
    )

    tracker.start_run(run_name="unit_test_run")
    tracker.log_params({"lr": 0.001, "batch_size": 16})
    tracker.log_metrics({"train_loss": 0.5}, step=1)

    dummy_artifact = tmp_path / "dummy.txt"
    dummy_artifact.write_text("checkpoint_data")
    tracker.log_artifact(dummy_artifact)

    tracker.end_run()

    assert (db_dir / "mlflow.db").exists()
