"""
Description:
    this is the module for creating util functions for project.

"""

# Futures
from __future__ import print_function

import datetime
import sys

sys.path.append('..')

# Built-in/Generic Imports
import os
from keras.callbacks import *
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle

__author__ = '{Chao ZHOU}'
__copyright__ = 'Copyright {02/01/2020}, {Variational autoencoder}'
__email__ = '{chaozhouucl@gmail.com}'
__status__ = '{dev_status}'


##############################################################################################################
# define project log path creation function
##############################################################################################################
# create project log path
def create_project_log_path(project_path, **kwargs):
    # year_month_day/hour_min/(model_log_dir, model_checkpoint_dir, tensorboard_log_dir)/
    date = datetime.datetime.now()

    program_day = project_path + 'Archive/' + date.strftime("%Y_%m_%d")
    if not os.path.exists(program_day):
        os.mkdir(program_day)
    readme = kwargs.pop('Readme')
    program_time = date.strftime("%H_%M_%S")
    for key, value in kwargs.items():
        program_time = program_time + '_' + key + '_{}'.format(value)

    program_log_parent_dir = os.path.join(program_day, program_time + '/')
    if not os.path.exists(program_log_parent_dir):
        os.mkdir(program_log_parent_dir)

    # model checkpoint dir
    model_checkpoint_dir = program_log_parent_dir + 'model_checkpoint_dir/'
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    # tensorboard_log_dir
    tensorboard_log_dir = program_log_parent_dir + 'tensorboard_log_dir/'
    if not os.path.exists(tensorboard_log_dir):
        os.mkdir(tensorboard_log_dir)

    # model_log_dir
    model_log_dir = program_log_parent_dir + 'model_log_dir/'
    if not os.path.exists(model_log_dir):
        os.mkdir(model_log_dir)

    # write exp log
    with open(program_log_parent_dir + 'Readme.txt', 'w') as f:
        f.write(readme + '\r\n')
        f.write('program log dir: ' + program_log_parent_dir)

    return model_checkpoint_dir, tensorboard_log_dir, model_log_dir


##############################################################################################################
# define functions that write/read data into/from pickle
##############################################################################################################
# define data writer func
def write_data(data, file_path):
    file_writer = open(file_path, 'wb')
    pickle.dump(data, file_writer)
    file_writer.close()


# define data reader func
def read_data(file_path):
    file_reader = open(file_path, 'rb')
    data = pickle.load(file_reader)
    file_reader.close()
    return data