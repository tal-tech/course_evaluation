#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Author: Harvey
Date: 07/12/2018
Modified: 09/18/2018
Including: three wrap-up function for all needed feature
functions following layered structure
Update: add the json file basic feature extractor
'''

import sys
import pandas as pd
from itertools import combinations
import numpy as np
import re
import json

sys.path.append('..')
from features.basicFeatures import *
from features.transFeatures import *
from features.crossFeatures import *
from util.jsonParser import jsonParser

# initialize work space for all feature functions
def initialWorkspace(configObj, file_name):
    # this part needed to be added one by one
    cache = dict()
    # load the dictionary into work space
    init_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../init/data')
    load_keyword_dict(init_data_path, cache)
    # lines below are only for ocean project
    if 'subjectJsonPath' in configObj['wordInput']:
        with open(configObj['wordInput']['subjectJsonPath']) as temp_json:
            subject_json = json.load(temp_json)
        temp_json.close()
        subject_chn = dict(list(zip(range(1,13),['english','chinese','math','chemistry','physics','biology','geography','politics','history','science','art','other'])))
        subject_dict = dict([(int(x['id']), subject_chn[int(x['subject'])]) for x in subject_json])
        if int(re.sub('.xls|.xlsx|.csv','', file_name)) not in subject_dict.keys():
            cache['subject'] = 'error'
        else:
            cache['subject'] = subject_dict[int(re.sub('.xls|.xlsx|.csv','', file_name))]
    return cache

# generate basic features from excel data    
def getBasicFeatures(configObj, file_name):
    cache = initialWorkspace(configObj, file_name)
    txt_file_path = configObj['input']['path'] + file_name
    # Here we suppose files are in the same directory
    wav_file_path = configObj['input']['path'] + file_name.split('.')[-2] + '.wav'
    result = []
    basic_features = [key for key, value in configObj['basicFeatures'].items() if int(value) == 1]
    result_column = []
    if len(basic_features) == 0:
        return pd.Series(), cache.copy()
    df = openFile(txt_file_path, cache)
    # Here we make the justification
    # if np.sum(df['text'].notnull()) == 0:
    if np.sum(df['textLength']) == 0:
        # at this line we need to generate the feature names by our selves
        stat_features = [x for x in ['getSentLen','getCharNum','getSentSpeed'] if x in basic_features]
        basic_features = [x for x in basic_features if x not in stat_features]
        stat_features = [y.split('get')[1]+'_'+x for x in ['Avg','Std','Min','25Per','50Per','75Per','Max','Var'] for y in stat_features]
        result_column = [x.split('get')[1] for x in basic_features] + stat_features
        return pd.Series(index = result_column), cache.copy()
    if 'getFileLen' in basic_features:
        result += [getFileLen(wav_file_path, cache)]
        result_column += ['FileLen']
    if 'getSpeedByFile'in basic_features:
        result += [getSpeedByFile(df,wav_file_path,cache)]
        result_column += ['SpeedByFile']
    if 'getVoiceOverFilePercent' in basic_features:
        result += [getVoiceOverFilePercent(df, wav_file_path, cache)]
        result_column += ['VoiceOverFilePercent']
    if 'getTotalXNum' in basic_features:
        wordList = getWordList(configObj['wordInput']['wordListPath'])
        result += [getTotalXNum(df, wordList, cache)]
        result_column += ['TotalXNum']
    if 'getTotalPauseWordNum' in basic_features:
        wordList = getWordList(configObj['wordInput']['stopWordsPath'])
        result += [getTotalPauseWordNum(df, wordList, cache)]
        result_column += ['TotalPauseWordNum']
    if 'getTotalPosNum' in basic_features:
        wordList = getWordList(configObj['wordInput']['stopWordsPath'])
        result += [getTotalPosNum(df, wordList, cache)]
        result_column += ['TotalPosNum']
    if 'getTotalNegNum' in basic_features:
        wordList = getWordList(configObj['wordInput']['stopWordsPath'])
        result += [getTotalNegNum(df, wordList, cache)]
        result_column += ['TotalNegNum']
    if 'getAliLenPercent' in basic_features:
        result += [getAliLenPercent(df, wav_file_path, cache)]
        result_column += ['AliLenPercent']
    # Here we group up the three statistic features
    stat_features = [x for x in ['getSentLen','getCharNum','getSentSpeed'] if x in basic_features]
    for func in stat_features:
        temp_result = eval(func + '(df,cache)')
        result += [np.mean(temp_result),np.std(temp_result)] + list(pd.Series(temp_result).describe())[3:] + [np.var(temp_result)]
        result_column += [func.split('get')[1]+'_'+x for x in ['Avg','Std','Min','25Per','50Per','75Per','Max','Var']]
    subjectword_features = [x for x in ['getSubjectWordFirstTime','getSubjectWordLastTime','getSubjectWordMaxDistance','getSubjectWordDensity'] if x in basic_features]
    for func in subjectword_features:
        subject = cache['subject']
        result += [eval(func + '(df, "/init/data", subject, cache)')]
        result_column += [func.split('get')[1]]
    keyword_features = [x for x in ['getNoteWordMaxDistance','getPraiseSentCount','getGreetSentCount'] if x in basic_features]
    for func in keyword_features:
        result += [eval(func + '(df, "/init/data", cache)')]
        result_column += [func.split('get')[1]]
    exclude_features = ['getFileLen','getSpeedByFile','getVoiceOverFilePercent','getTotalXNum','getTotalPauseWordNum','getTotalPosNum','getTotalNegNum','getAliLenPercent'] + stat_features + subjectword_features + keyword_features
    basic_features = [x for x in basic_features if x not in exclude_features]
    for func in basic_features:
        result += [eval(func + '(df, cache)')]
    result_column += [x.split('get')[1] for x in basic_features]
    return pd.Series(result, index=result_column), cache.copy()

# generate transformed basic features from basic feature results
def getTransFeatures(configObj, basic_feature_df, cache):
    trans_features = pd.Series([key for key, value in configObj['transFuncs'].items() if int(value) == 1])
    if len(trans_features) == 0:
        return pd.Series()
    trans_columns = [x+'_'+y.split('get')[1] for y in trans_features for x in basic_feature_df.index]
    # add one more line to justify whether all elements in basic features are NA
    if np.sum(basic_feature_df.notnull()) == 0:
        return pd.Series(index = trans_columns)
    trans_result = []
    for func in trans_features:
        trans_result += [eval(func+'(basic_feature_df)')]
    # trans_result = pd.concat(trans_result)
    # trans_columns = [x+'_'+y.split('get')[1] for y in trans_features for x in basic_feature_df.index]
    return pd.Series(list(pd.concat(trans_result)), index = trans_columns)

# generate crossed basic features from basic feature results
def getCrossetCrossFeatures(configObj, basic_feature_df, cahce):
    cross_features = [item[0] for item in configObj['crossFuncs'].items() if len(item[1]) !=0]
    cross_result = []
    cross_column = []
    for func in cross_features:
        feature_pairs = configObj['crossFuncs'][func]
        if feature_pairs == ['ALL'] or feature_pairs == ['All']:
            basic_features = [key.split('get')[1] for key, value in configObj['basicFeatures'].items() if int(value) == 1]
            basic_features = [x for x in basic_features if x not in ['SentLen','CharNum','SentSpeed']]
            feature_pairs = list(combinations(basic_features, 2))
        for pair in feature_pairs:
            cross_result += [eval('get'+func+'(basic_feature_df["'+pair[0]+'"],basic_feature_df["'+pair[1]+'"])')]
            cross_column += [pair[0]+'_'+pair[1]+'_'+func]
    if np.sum(basic_feature_df.notnull()) == 0:
        return pd.Series(index=cross_column)
    return pd.Series(list(cross_result), index=cross_column)

# generate sentence level features from excel file [only for DSSM]
def getSentenceFeatures(configObj, file_name):
    txt_file_path = configObj['input']['path'] + file_name
    cache = {}
    result = []
    basic_features = [key for key, value in configObj['basicFeatures'].items() if int(value) == 1]
    result_column = []
    df = openFile(txt_file_path, cache)
    # filter out the featurs that we can do nothing with
    available_sentence_feature = ['getSentLen','getCharNum','getSentSpeed','getDuplicate1Abs','getDuplicate1Percent',
    'getDuplicate2Abs','getDuplicate2Percent','getTotalQuestionSentNum','getTotalNounNum','getTotalVerbNum','getTotalAdjNum',
    'getTotalAdvNum','getTotalXNum','getTotalPauseWordNum']
    basic_features = list(set(basic_features).intersection(set(available_sentence_feature)))
    # change some string part of basic_features
    basic_features = [x.replace('Total','')+ 'Vector' for x in basic_features]
    # after we do this we can run the basic features one by one
    for func in basic_features:
        result_column += [re.sub('get|Vector', '', func)]
        if np.sum(df['text'].notnull()) == 0:
            continue
        if func == 'getPauseWordNumVector':
            wordList = getWordList(configObj['wordInput']['stopWordsPath'])
            result += [eval(func + "(df, wordList, cache)")]
        elif func == 'getXNumVector':
            wordList = getWordList(configObj['wordInput']['wordListPath'])
            result += [eval(func + "(df, wordList, cache)")]
        else:
            result += [eval(func + '(df, cache)')]
    result = np.array(result).T
    return result, result_column

# generate timestamp format features from excel file
def getTimsStampFeatures(configObj, file_name):
    cache = initialWorkspace(configObj, file_name)
    txt_file_path = configObj['input']['path'] + file_name
    result = []
    timestamp_features = [key for key, value in configObj['timestampFeatures'].items() if int(value) == 1]
    result_column = []
    if len(timestamp_features) == 0:
        return pd.Series()
    df = openFile(txt_file_path, cache)
    stats_ts_features = [x for x in timestamp_features if 'Stats' in x]
    # get ready for the no length df
    if np.sum(df['text'].notnull()) == 0:
        result_column = [re.sub('^get','',x) for x in timestamp_features if x not in stats_ts_features]
        stats_columns = ['Count', 'Mean', 'Std', 'Variance', 'Skew', 'Kurtosis', 'Range', '25Per','50Per','75Per']
        result_column += [re.sub('^get','',y) + '_' + x for x in stats_columns for y in stats_ts_features]
        return pd.Series(index=result_column)
    # add few lines to get the timeseries stats
    for func in stats_ts_features:
        stats_result, stats_columns = eval(func + "(df, cache)")
        result += stats_result
        result_column += [re.sub('^get','',func) + '_' + x for x in stats_columns]
    no_dict_features = [x for x in ['getLongBlankTimestamp', 'getQuestionTimestamp', 'getTalkTurnTimestamp'] if x in timestamp_features]
    for func in no_dict_features:
        result += [eval(func + "(df, cache)")]
        result_column += [re.sub('^get','',func)]
    timestamp_features = [x for x in timestamp_features if x not in no_dict_features+stats_ts_features]
    for func in timestamp_features:
        if func == 'getSubjectWordTimestamp':
            subject = cache['subject']
            result += [eval(func + '(df, "/init/data", subject, cache)')]
        else:
            result += [eval(func + '(df, "/init/data", cache)')]
        result_column += [re.sub('^get','',func)]
    return pd.Series(result, index=result_column)

# generate basic features from json data
def getBasicFeaturesFromjson(configObj, json_str):
    cache = initialWorkspace(configObj, None)
    basic_features = [key for key, value in configObj['basicFeatures'].items() if int(value) == 1]
    df = jsonParser(configObj, json_str)
    if configObj['input']['object'] == 'student':
        cache['fileLen'] = int(json_str['student']['duration'])/1000
    else:
        cache['fileLen'] = int(json_str['teacher']['duration'])/1000
    result = []
    result_column = []
    # if np.sum(df['text'].notnull()) == 0:
    if np.sum(df['textLength']) == 0:
        # at this line we need to generate the feature names by our selves
        stat_features = [x for x in ['getSentLen','getCharNum','getSentSpeed'] if x in basic_features]
        basic_features = [x for x in basic_features if x not in stat_features]
        stat_features = [y.split('get')[1]+'_'+x for x in ['Avg','Std','Min','25Per','50Per','75Per','Max','Var'] for y in stat_features]
        result_column = [x.split('get')[1] for x in basic_features] + stat_features
        return pd.Series(index = result_column), cache.copy()
    if 'getFileLen' in basic_features:
        result += [getFileLen(None, cache)]
        result_column += ['FileLen']
    if 'getTotalXNum' in basic_features:
        wordlist_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', configObj['wordInput']['wordListPath'])
        wordList = getWordList(wordlist_path)
        result += [getTotalXNum(df, wordList, cache)]
        result_column += ['TotalXNum']
    if 'getTotalPauseWordNum' in basic_features:
        wordlist_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', configObj['wordInput']['stopWordsPath'])
        wordList = getWordList(wordlist_path)
        result += [getTotalPauseWordNum(df, wordList, cache)]
        result_column += ['TotalPauseWordNum']
    fileLenFeature = [x for x in ['getSpeedByFile','getVoiceOverFilePercent','getAliLenPercent'] if x in basic_features]
    for func in fileLenFeature:
        result += [eval(func + '(df, None, cache)')]
        result_column += [func.split('get')[1]]
    stat_features = [x for x in ['getSentLen','getCharNum','getSentSpeed'] if x in basic_features]
    for func in stat_features:
        temp_result = eval(func + '(df,cache)')
        result += [np.mean(temp_result),np.std(temp_result)] + list(pd.Series(temp_result).describe())[3:] + [np.var(temp_result)]
        result_column += [func.split('get')[1]+'_'+x for x in ['Avg','Std','Min','25Per','50Per','75Per','Max','Var']]
    subjectword_features = [x for x in ['getSubjectWordFirstTime','getSubjectWordLastTime','getSubjectWordMaxDistance','getSubjectWordDensity'] if x in basic_features]
    for func in subjectword_features:
        subject = json_str['class']['subject']
        result += [eval(func + '(df, "/init/data", subject, cache)')]
        result_column += [func.split('get')[1]]
    keyword_features = [x for x in ['getNoteWordMaxDistance','getPraiseSentCount','getGreetSentCount'] if x in basic_features]
    for func in keyword_features:
        result += [eval(func + '(df, "/init/data", cache)')]
        result_column += [func.split('get')[1]]
    exclude_features = ['getFileLen','getSpeedByFile','getVoiceOverFilePercent','getTotalXNum','getTotalPauseWordNum','getTotalPosNum','getTotalNegNum','getAliLenPercent'] + stat_features + subjectword_features + keyword_features
    basic_features = [x for x in basic_features if x not in exclude_features]
    for func in basic_features:
        result += [eval(func + '(df, cache)')]
    result_column += [x.split('get')[1] for x in basic_features]
    return pd.Series(result, index=result_column), cache.copy()

# generate timestamp format features from json data
def getTimsStampFeaturesFromjson(configObj, json_str):
    cache = initialWorkspace(configObj, None)
    subject_chn = dict(list(zip(range(1,13),['english','chinese','math','chemistry','physics','biology','geography','politics','history','science','art','other'])))
    cache['subject'] = subject_chn[json_str['class']['subject']]
    result = []
    result_column = []
    timestamp_features = [key for key, value in configObj['timestampFeatures'].items() if int(value) == 1]
    if len(timestamp_features) == 0:
        return pd.Series()
    df = jsonParser(configObj, json_str)
    stats_ts_features = [x for x in timestamp_features if 'Stats' in x]
    if np.sum(df['text'].notnull()) == 0:
        result_column = [re.sub('^get','',x) for x in timestamp_features if x not in stats_ts_features]
        stats_columns = ['Count', 'Mean', 'Std', 'Variance', 'Skew', 'Kurtosis', 'Range', '25Per','50Per','75Per']
        result_column += [re.sub('^get','',y) + '_' + x for x in stats_columns for y in stats_ts_features]
        return pd.Series(index=result_column)
    # add few lines to get the timeseries stats
    for func in stats_ts_features:
        stats_result, stats_columns = eval(func + "(df, cache)")
        result += stats_result
        result_column += [re.sub('^get','',func) + '_' + x for x in stats_columns]
    no_dict_features = [x for x in ['getLongBlankTimestamp', 'getQuestionTimestamp', 'getTalkTurnTimestamp'] if x in timestamp_features]
    for func in no_dict_features:
        result += [eval(func + "(df, cache)")]
        result_column += [re.sub('^get','',func)]
    timestamp_features = [x for x in timestamp_features if x not in no_dict_features+stats_ts_features]
    for func in timestamp_features:
        if func == 'getSubjectWordTimestamp':
            subject = cache['subject']
            result += [eval(func + '(df, "/init/data", subject, cache)')]
        else:
            result += [eval(func + '(df, "/init/data", cache)')]
        result_column += [re.sub('^get','',func)]
    return pd.Series(result, index=result_column)
