import logging


def create_logger(debug=False, path=None):
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.setLevel(logging.DEBUG) if debug else logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(path) if path else logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
