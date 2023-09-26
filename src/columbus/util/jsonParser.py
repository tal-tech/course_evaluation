#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Author: Harvey
Date: 09/17/2018
The program used for parse the json string passed by C++ prgram
'''

import pandas as pd
import re
import numpy as np
import json
import logging
import traceback

def jsonParser(configObj, json_str):
    try:
        jsonStr = pd.DataFrame(json_str[configObj['input']['object']]['text'])
        df = pd.DataFrame(jsonStr)
        df.drop_duplicates(['begin_time', 'end_time', 'text'],
                         inplace=True)
        # add two lines to convert all the timestamp into int
        df['begin_time'] = df['begin_time'].apply(lambda x: int(x))
        df['end_time'] = df['end_time'].apply(lambda x: int(x))
        df['status_code'] = 0
        # double check replace na with string value
        class_begin_at = int(json_str['class']['first_start_at']) if len(
            str(json_str['class']['first_start_at'])) == 13 else int(
                json_str['class']['first_start_at']) * 1000
        class_end_at = int(json_str['class']['last_end_at']) if len(
            str(json_str['class']['last_end_at'])) == 13 else int(
                json_str['class']['last_end_at']) * 1000
        video_begin_at = int(
            json_str[configObj['input']['object']]['start_time_ms'])
        video_begin_at = video_begin_at if len(
            str(video_begin_at)) == 13 else video_begin_at * 1000
        newDf = df[df.text != ''].copy()
        index = []
        index.extend((newDf.begin_time).tolist())
        blankStart = [0] 
        blankStart.extend((newDf.iloc[:-1].end_time).tolist())
        blankStart = np.array(blankStart)
        blankEnd = newDf.begin_time
        blank = [{
            "begin_time": x[0],
            "end_time": x[1],
            "status_code": 0,
            "text": ""
        } for x in zip(blankStart, blankEnd)]
        df_blank = pd.DataFrame(blank)
        newMerge = pd.concat([df_blank, newDf])
        newMerge.sort_values('begin_time', inplace=True)
        newMerge['begin_time'] = newMerge['begin_time'] + video_begin_at
        newMerge['end_time'] = newMerge['end_time'] + video_begin_at
        
        newMerge = newMerge[(newMerge.begin_time >= class_begin_at)
                            & (newMerge.end_time <= class_end_at)]
        newMerge['sentence_id'] = range(1, newMerge.shape[0] + 1)
        newMerge['timeLength'] = newMerge.end_time - newMerge.begin_time
        # newMerge = newMerge[newMerge['timeLength'] >= 0].copy()
        newMerge['textLength'] = newMerge.text.apply(
            lambda x: len(re.sub(r'[^\w]|_', '', str(x))))
        newMerge = newMerge[[
            'sentence_id', 'begin_time', 'end_time', 'timeLength',
            'textLength', 'text'
        ]]
        newMerge.index = range(1, newMerge.shape[0] + 1)
    except:
        newMerge = pd.DataFrame(columns=[
            'sentence_id', 'begin_time', 'end_time', 'timeLength',
            'textLength', 'text'
        ])
        logging.error('Error parser jsonStr,detail is \n{}'.format(
            traceback.format_exc()))
    return newMerge