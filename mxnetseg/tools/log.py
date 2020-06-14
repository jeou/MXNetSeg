# coding=utf-8

import logging

__all__ = ['get_logger']

_log_level = {
    'notset': logging.NOTSET,
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}


def get_logger(name: str = 'seg_logger', level: str = 'info', log_file: str = None):
    """
    logger

    :param name: logger name
    :param level: log level
    :param log_file: file path
    :return: logging.logger
    """
    # logger
    level = _log_level[level]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s, %(filename)s(line:%(lineno)d), [%(levelname)s]:: %(message)s')

    # stream
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # file
    if log_file:
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
