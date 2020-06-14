# coding=utf-8

import os
import platform


def data_dir():
    """data directory in the filesystem for model storage, for example when downloading models"""
    return os.getenv('MXNET_HOME', _data_dir_default())


def _data_dir_default():
    """default data directory depending on the platform and environment variables"""
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'mxnet')
    else:
        return os.path.join(os.path.expanduser("~"), '.mxnet')


def root_path():
    """root path"""
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return root_dir


def demo_path():
    """demo path"""
    return os.path.join(root_path(), 'demo')


def dataset_path():
    """dataset path"""
    return os.path.join(root_path(), 'dataset')


def records_path():
    """records path"""
    return os.path.join(root_path(), 'records')


def record_path(model_name: str):
    """record path for a specific model"""
    path = os.path.join(records_path(), model_name.lower())
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def weights_path():
    """models params path."""
    return os.path.join(root_path(), 'weights')


def weight_path(model_name: str):
    """model params path for a specific model"""
    path = os.path.join(weights_path(), model_name.lower())
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def logs_path():
    """
    logs path for fitlog.
    better keep consistent with /.fitlog/.fitconfig.
    """
    return os.path.join(root_path(), 'logs')
