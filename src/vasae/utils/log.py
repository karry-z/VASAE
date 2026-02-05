import logging
import os


def get_logger():
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger()
    logger.info(f">>> start logging")
    return logger
