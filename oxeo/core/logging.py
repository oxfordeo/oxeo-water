import logging


def get_logger() -> logging.Logger:
    """Initializes standard Logger that Prefect Cloud can pick up."""

    logger = logging.getLogger("oxeo")
    if len(logger.handlers) == 0:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)8s | %(module)s:%(funcName)s:%(lineno)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


def set_level(level: int = 40) -> None:
    """Defaults to ERROR, useful to run in notebooks."""
    get_logger().setLevel(level)
