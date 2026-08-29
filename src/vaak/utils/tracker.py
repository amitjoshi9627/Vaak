from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """Experiment tracker wrapping MLflow APIs with SQLite backend support."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | Path = "sqlite:///mlruns/mlflow.db",
    ) -> None:
        if isinstance(tracking_uri, Path):
            if tracking_uri.is_dir() or tracking_uri.suffix != ".db":
                db_path = tracking_uri / "mlflow.db"
            else:
                db_path = tracking_uri
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.tracking_uri = f"sqlite:///{db_path.resolve()}"
        else:
            self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)

        # Handle soft-deleted experiments automatically
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is not None and experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None) -> str:
        """Start a new MLflow tracking run and return its ID."""
        run = mlflow.start_run(run_name=run_name)
        return str(run.info.run_id)

    def end_run(self) -> None:
        """End the active MLflow tracking run."""
        if mlflow.active_run() is not None:
            mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log configuration parameters and hyperparameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics at a specific step or epoch."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: Path) -> None:
        """Log a file artifact such as a checkpoint or report."""
        if local_path.exists():
            mlflow.log_artifact(str(local_path))
