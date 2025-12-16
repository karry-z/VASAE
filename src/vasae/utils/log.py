import logging
import os


def get_logger(logpath):
    logpath = os.path.abspath(logpath)
    logname = os.path.splitext(os.path.basename(logpath))[0]
    if not ".log" in logpath:
        current_pyfile_path = logpath
        logdir = os.path.join(os.path.dirname(current_pyfile_path), "logs")
        os.makedirs(logdir, exist_ok=True)

        logpath = os.path.join(logdir, f"{logname}.log")

    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(logpath)],
    )
    logger = logging.getLogger(logname)
    logger.info(f">>> start logging at {logpath}")
    return logger
