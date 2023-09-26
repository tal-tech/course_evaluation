import os
import json
import logging
import numpy as np
import pandas as pd
import warnings
import configparser
from logging.config import fileConfig
import re
import traceback
from Ticktock import ticktock

warnings.filterwarnings("ignore")

basePath = os.path.dirname(os.path.realpath(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(basePath, '../../config/algorithms.ini'))
'''
读取日志配置
'''
fileConfig(os.path.join(basePath, '../../config/logging_config.ini'), disable_existing_loggers=False)
logger = logging.getLogger()
'''
读取状态码
'''
default_error_code = int(config['Output'].get('default_error_code'))
default_error_message = config['Output'].get('default_error_message')
normal = int(config['ErrorCode'].get('normal'))
success = int(config['ErrorCode'].get('success'))
input_error = int(config['ErrorCode'].get('input_error'))
text_empty = int(config['ErrorCode'].get('text_empty'))
keyword_list_empty = int(config['ErrorCode'].get('keyword_list_empty'))
columbus_error = int(config['ErrorCode'].get('columbus_error'))
class_evaluation_error = int(config['ErrorCode'].get('class_evaluation_error'))
jabber_error = int(config['ErrorCode'].get('jabber_error'))
'''
设置全局变量
'''

re_question = re.compile('\?|？')
re_no_char = compiled_rule = re.compile(r'[^\w]|_')

multi_item_2_fun = {'speed': 'get_speak_speed_s', 'sent_num': 'get_sentence_num',
                    'sent_av_len': 'get_sentence_average_length', 'punch_num': 'get_pause_words_num',
                    'question_num': 'get_question_num',
                    'short_sent_num': 'get_short_sentence_num', 'mid_sent_num': 'get_median_sentence_num',
                    'long_sent_num': 'get_long_sentence_num'}
