import logging


def get_logger() -> logging.Logger:
    """Initializes standard Logger that Prefect Cloud can pick up."""

    logging.basicConfig(
        format="%(asctime)s | %(levelname)8s | %(module)s:%(funcName)s:%(lineno)s - %(message)s",
        level=logging.DEBUG,
    )
    logger = logging.getLogger("oxeo.water")
    return logger


logger = get_logger()
