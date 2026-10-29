import logging

global_logger = None
dup_filter = None


class DuplicateFilter(object):
    """
    For filtering specific messages
    """
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        """
        Determining filtering or not
        Args:
            record:

        Returns: boolean

        """
        rv = record.msg not in self.msgs
        if getattr(record, "log_once", False):
            self.add_msg(record.msg)
        return rv

    def add_msg(self, msg):
        """
        Add a msg to the filter
        Args:
            msg: message for filtering

        Returns: None

        """
        self.msgs.add(msg)

    def reset(self):
        """
        Reset the filter
        Returns: None

        """
        self.msgs.clear()


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


def get_logger():
    """
    Get the global logger
    Args:

    Returns: None

    """
    global global_logger
    global dup_filter
    if global_logger is None:
        dup_filter = DuplicateFilter()
        logger = logging.getLogger("MetaDrive")
        logger.propagate = False
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        # create formatter and add it to the handlers
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
        logger.addFilter(dup_filter)
        global_logger = logger
    return global_logger


def set_propagate(propagate=False):
    global global_logger
    global_logger.propagate = propagate


def set_log_level(level=logging.INFO):
    """
    Set the level of global logger
    Args:
        level: level.INFO, level.DEBUG, level.WARNING

    Returns: None

    """
    global global_logger
    if global_logger.level != level:
        global_logger.setLevel(level)


def reset_logger():
    """
    Reset the logger
    Returns: None
    """
    global dup_filter
    dup_filter.reset()
