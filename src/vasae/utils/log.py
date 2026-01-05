import logging
import os


def get_logger(logpath):
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(logpath))],
    )
    logger = logging.getLogger(str(logpath))
    logger.info(f">>> start logging at {logpath}")
    return logger
