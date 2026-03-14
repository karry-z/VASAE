import logging


def get_logger(name: str = "vasae"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(levelname)s][%(asctime)s][%(filename)s:%(funcName)s] %(message)s",
                datefmt="%Y%m%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
