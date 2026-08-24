from pathlib import Path

from vaak.config.settings import load_config
from vaak.core.logging import configure_logging, get_logger


def main() -> None:
    configure_logging()

    logger = get_logger(__name__)

    config = load_config(Path("configs/config.yaml"))

    logger.info(
        f"Vaak initialized | version={config.version} | environment={config.environment}"
    )


if __name__ == "__main__":
    main()
