import logging

global_logger = None


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "[%(levelname)s] %(message)s (%(name)s %(filename)s:%(lineno)d)"
    format = "[%(levelname)s] %(message)s (%(filename)s:%(lineno)d)"
    simple_format = "[%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + simple_format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(level=logging.INFO, propagate=False):
    global global_logger
    if global_logger is None:
        logger = logging.getLogger("MetaDrive")
        logger.propagate = propagate
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        # create formatter and add it to the handlers
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
        global_logger = logger
    if global_logger.level != level:
        global_logger.setLevel(level)
    return global_logger
