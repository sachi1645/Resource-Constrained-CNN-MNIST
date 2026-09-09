import logging
from pathlib import Path


def get_logger(name="mnist_cnn", log_file=None):
    """
    Create and return a project logger.

    Parameters
    ----------
    name : str
        Name of the logger.
    log_file : str or None
        Optional path for saving logs to a file.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if the logger is created again.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger