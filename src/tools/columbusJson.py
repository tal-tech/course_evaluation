#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Author: Harvey
Date: 08/25/2018
Modified: 08/02/2018
Including: the Json SDK for Columbus project
Please follow the help guidance running the SDK
'''

import pandas as pd
import json
from columbus.util.wrapFeature import *
from init import readConfig
import logging
import warnings
import getopt
import multiprocessing
import traceback

def columbusJsonSDK(configFile, jsonStr,task_id=''):
    logging.info('start columbusJson SDK work...')
    try:
        configObj = readConfig(configFile)
    except:
        logging.error('failed on load configration file, detail is {}'.format(traceback.format_exc()))
        return pd.DataFrame()
    try:
        basic_feature_df, _cache = getBasicFeaturesFromjson(configObj, jsonStr)
    except:
        logging.error('failed on basic feature step, detail is {}'.format(traceback.format_exc()))
        return pd.DataFrame()
    if 'transFuncs' in configObj.keys():
        try:
            trans_feature_df = getTransFeatures(configObj, basic_feature_df, _cache)
        except:
            logging.error('failed on trans feature step, detail is {}'.format(traceback.format_exc()))
            return pd.DataFrame()
    else:
        trans_feature_df = pd.Series()
    if 'crossFuncs' in configObj.keys():
        try:
            cross_feature_df = getCrossetCrossFeatures(configObj, basic_feature_df, _cache)
        except:
            logging.error('failed on cross feature step, detail is {}'.format(traceback.format_exc()))
            return pd.DataFrame()
    else:
        cross_feature_df = pd.Series()
    if 'timestampFeatures' in configObj.keys():
        try:
            timestamp_feature_df = getTimsStampFeaturesFromjson(configObj, jsonStr)
        except:
            logging.error('failed on timestamp info step, detail is {}'.format(traceback.format_exc()))
            return pd.DataFrame()
    else:
        timestamp_feature_df = pd.Series()
    file_name_df = pd.Series([jsonStr['class']['id']], index=['classID'])
    allFeatures = pd.concat([file_name_df, basic_feature_df, trans_feature_df, cross_feature_df, timestamp_feature_df])
    # the fillna step could be dangerous since we may change some feature
    allFeatures = allFeatures.fillna(0)
    logging.info('columbusSDK work succeed!')
    return pd.DataFrame([allFeatures])

if __name__ == '__main__':
    # ignore the warnings
    warnings.filterwarnings("ignore")
    # set up the log directory 
    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log/columbus.log')
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
    )
    # get the arguments from terminal
    opts, args = getopt.getopt(sys.argv[1:], 'hc:i:o:', ["help","config=","inputPath=","outputPath="])
    config_file = None
    temp_inpath = None
    temp_outpath = None
    for key, value in opts:
        if key in ['-h', '--help']:
            print('-h --help: Show function help; -c --config: Configuration file')
            sys.exit(0)
        if key in ['-c','--config']:
            config_file = value
        if key in ['-i','--inputPath']:
            temp_inpath = value
        if key in ['-o','--outputPath']:
            temp_outpath = value
    if config_file is None or temp_inpath is None or temp_outpath is None:
        print('Missing configuration file or input path or output path, please ask FeatureExtract.py --help[-h] for more help!')
        sys.exit(0)
    json_list = [json.load(open(os.path.join(temp_inpath, x),'r')) for x in os.listdir(temp_inpath) if '.json' in x]
    
    print('precess start')
    pool = multiprocessing.Pool(processes = 40)
    feature_result_tmp = []
    for x in json_list:
        feature_result_tmp.append(pool.apply_async(columbusJsonSDK, (config_file, x)))
    pool.close()
    pool.join()
    print('all process done')
    result = []
    for x in feature_result_tmp:
        result.append(x.get())
    feature_result = pd.concat(result, axis=0)
    feature_result.to_csv(os.path.join(temp_outpath, 'feature_result.csv'), index = False)
    print('Jobs done!')