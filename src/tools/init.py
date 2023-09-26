#!/usr/bin/python
# -*- coding: utf-8 -*-


'''
Project 'Columbus' configuration file parser
author: Jiahao Chen
Date: July 11, 2018
Modified: 08/02/2018
解析配置文件，并返回dict形式结果
'''
import configparser

def readConfig(config_file):
    config=configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file, encoding="utf-8-sig")
    config_dic=dict(config)
    for section in config_dic:
        section_content=dict(config_dic[section])
        # print(section_content)
        if section in ['basicFeatures', 'transFuncs', 'timestampFeatures']:
            section_content={'get{}'.format(x[0]):x[1] for x in section_content.items()}
        # if section in ['wordInput', 'crossFuncs']:
        if section == 'crossFuncs':
            section_content={x[0]:eval(x[1]) for x in section_content.items()}
        config_dic[section]=section_content
    return config_dic