import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=level,
        format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""

    return logging.getLogger(name)
