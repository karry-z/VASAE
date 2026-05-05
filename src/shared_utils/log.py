import logging
import time


def get_logger(name: str | None = None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] %(message)s",
            datefmt="%Y%m%d %H:%M:%S UTC",
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
