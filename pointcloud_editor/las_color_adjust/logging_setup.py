"""Per-module logging setup."""
import logging


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create a per-module logger with console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent child loggers from duplicating messages to parent handlers
        logger.propagate = False
    return logger
