#!/usr/bin/python
#encoding:utf-8
'''
Project 'Columbus' basic feature function modules
author: Guowei Xu
Date: July 11, 2018
Modified: 08/02/2018
所有函数返回值要么是单个变量，要么为一个numpy array
'''
import pandas as pd
import numpy as np
import wave
import jieba
import requests
import json
import jieba.posseg as pseg
import jieba
import time
import re
import os
from scipy.stats import skew
from scipy.stats import kurtosis
# jieba.enable_parallel()
#cache是一个全局变量贮存已经计算过的数值，避免重复运算
def openFile(path, cache):
    df = pd.read_excel(path)
    df['text'] = df['text'].fillna('')
    cache['df'] = df
    return df

#将句子拼接并返回所有文字
def getText(df, cache):
    text = ''
    for t in df['text']:
         if(isinstance(t, str)):#检查是否为string，排除nan
            text = text+'\n'+t.replace(" ", '')
    cache['text'] = text
    return text



#说话时长(秒)
def getVoiceLen(df, cache):
    # voiceLen = float(sum(df['timeLength']))/1000
    voiceLen = float(np.sum(df[df['textLength']>0]['timeLength']))/1000
    cache['voiceLen'] = voiceLen
    return voiceLen



#文件时长
####################################################3
######### This function  is from 心汝
#####################################################
def getFileLen(path_to_wav, cache):
    if 'fileLen' in cache.keys():
        return cache['fileLen']
    #open a wave file, and return a Wave_read object
    f = wave.open(path_to_wav,"rb")
    #read the wave's format infomation,and return a tuple
    params = f.getparams()
    #get the info
    framerate, nframes = params[2:4]
    fileLen = nframes * (1.0/framerate)
    cache['fileLen'] = fileLen
    return fileLen


#语音语速
def getSpeedByVoice(df, cache):
    totalCharNum, voiceLen = 0.0, 0.0
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if(not 'voiceLen' in cache.keys()):
        voiceLen = getVoiceLen(df, cache)
    else:
        voiceLen = cache['voiceLen']
    if(voiceLen != 0):
        speedByVoice = totalCharNum/voiceLen #总字数处以总说话时长
        cache['speedByVoice'] = speedByVoice
    else:
        speedByVoice = 0
    return speedByVoice

#文件语速, 字数/文件时长
def getSpeedByFile(df, path_to_wav, cache):
    totalCharNum, fileLen = 0.0, 0.0
    if(not('totalCharNum' in cache.keys())):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if(not('fileLen' in cache.keys())):
        fileLen =  getFileLen(path_to_wav, cache)
    else:
        fileLen = cache['fileLen']
    return totalCharNum/fileLen


#总字数
def getTotalCharNum(df, cache):
    if not 'text' in cache.keys():
        text = getText(df,cache)
    else:
        text = cache['text']
    totalCharNum = float(len(re.findall('\w', str(text))))
    # totalCharNum = float(sum(df['textLength']))
    cache['totalCharNum'] = totalCharNum
    return totalCharNum


#有效说话百分比
def getVoiceOverFilePercent(df, path_to_wav, cache):
    fileLen, voiceLen = 0.0, 0.0
    if(not 'fileLen' in cache.keys()):
        fileLen = getFileLen(path_to_wav, cache)
    else:
        fileLen = cache['fileLen']
    if(not 'voiceLen' in cache.keys()):
        voiceLen = getVoiceLen(df, cache)
    else:
        voiceLen = cache['voiceLen']
    voiceOverFilePercent = float(voiceLen)/fileLen
    cache['voiceOverFilePercent'] = voiceOverFilePercent
    return voiceOverFilePercent


#总句数
def getTotalSentNum(df, cache):
    # return df['text'].shape[0]
    return np.sum(df['textLength']>0)


#平均句长
def getAvgCharPerSent(df, cache):
    totalCharNum, charNum = 0.0, 0.0
    if(not ('totalCharNum' in cache.keys() and 'charNum' in cache.keys())):
        totalCharNum= getTotalCharNum(df, cache)
        charNum = getCharNum(df, cache)
        avgCharPerSent = float(totalCharNum)/charNum.shape[0]
    else:
        avgCharPerSent = float(cache['totalCharNum'])/cache['charNum'].shape[0]
    cache['avgCharPerSent'] = avgCharPerSent
    return avgCharPerSent

#总疑问句
#目前是统计整个文本包含多少问号，方法比较原始，后期可做改进
def getTotalQuestionSentNum(df, cache):
    if not 'text' in cache.keys():
        text = getText(df,cache)
    else:
        text = cache['text']
    return text.count('？')+text.count('?')#这里两个问号不同，前一个事中文输入法后一个为英文输入法

#计算词性并把分词按照词性分为不同的list，放到cache里
def getPOS(text, cache):
    noun = []
    verb = []
    adj = []
    adv = []
    noun_flag = ['n', 's', 'nr', 'ns', 'nt', 'nw', 'nz', 'vn']
    verb_flag = ['v', 'vd', 'vn']
    adj_flag = ['a', 'ad', 'an']
    adv_flag = ['d']
    words = pseg.cut(text)
    for w in words:
        if(w.flag in noun_flag):
            noun.append(w)
        elif(w.flag in verb_flag):
            verb.append(w)
        elif(w.flag in adj_flag):
            adj.append(w)
        elif(w.flag in adv_flag):
            adv.append(w)
    cache['noun'] = noun
    cache['verb'] = verb
    cache['adj'] = adj
    cache['adv'] = adv
    return noun, verb, adj, adv

#总名词数量
def getTotalNounNum(df, cache):
    # changed to df
    if(not 'text' in cache['text']):
        text = getText(df,cache)
    else:
        text = cache['text']
    nounNum = 0
    if(not 'noun' in cache.keys()):
        noun, _, _, _ = getPOS(text, cache)
        nounNum = len(noun)
    else:
        nounNum = len(cache['noun'])
    return nounNum

#总动词数量
def getTotalVerbNum(df, cache):
    if(not 'text' in cache['text']):
        text = getText(df,cache)
    else:
        text = cache['text']
    verbNum = 0
    if(not 'verb' in cache.keys()):
        _, verb, _, _ = getPOS(text, cache)
        verbNum = len(verb)
    else:
        verbNum = len(cache['verb'])
    return verbNum

#总形容词数量
def getTotalAdjNum(df, cache):
    if(not 'text' in cache['text']):
        text = getText(df,cache)
    else:
        text = cache['text']
    adjNum = 0
    if(not 'adj' in cache.keys()):
        adj = getPOS(text, cache)
        adjNum = len(adj)
    else:
        adjNum = len(cache['adj'])
    return adjNum

#总副词数量
def getTotalAdvNum(df, cache):
    if(not 'text' in cache['text']):
        text = getText(df,cache)
    else:
        text = cache['text']
    advNum = 0
    if(not 'adv' in cache.keys()):
        adv = getPOS(text, cache)
        advNum = len(adv)
    else:
        advNum = len(cache['adv'])
    return advNum

#获取X出现的次数，需要提供包含X词的list X_list
def getTotalXNum(df, X_list, cache):
    target_pattern = re.compile('|'.join(X_list))
    totalXNum = np.zeros(len(X_list))
    text = ""
    if(not 'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    totalXNum = len(target_pattern.findall(text))
    cache['totalXNum'] = totalXNum
    return totalXNum

# #以滑窗的形式遍历text，统计某个词x出现的次数，滑窗长度和x一样
# def countXNum(text, x):
#     count = 0
#     windowSize = len(x)
#     for i in range(0, len(text)-windowSize+1):
#         if(text[i:i+windowSize] == x):
#             count += 1
#     return count



#正情感词数目
def getTotalPosNum(df, pauseword, cache):
    if(not 'totalPosNum' in cache.keys()):
        totalPosNum, totalNegNum = sentimentJudger(df, pauseword, cache)
        cache['totalPosNum'] = totalPosNum
        cache['totalNegNum'] = totalNegNum
        return totalPosNum
    else:
        return cache['totalPosNum']

#负情感词数目
def getTotalNegNum(df, pauseword, cache):
    if(not 'totalNegNum' in cache.keys()):
        totalPosNum, totalNegNum = sentimentJudger(df, pauseword, cache)
        cache['totalPosNum'] = totalPosNum
        cache['totalNegNum'] = totalNegNum
        return totalNegNum
    else:
        return cache['totalNegNum']


#统计词语的情感，忽略停顿词
def sentimentJudger(df, pauseword, cache):
    text = ""
    pos_count = 0
    neg_count = 0
    AK = 'ONltuGBOpjucDTBrO1XlkKK9'
    SK = 'aPkeHTSHk0LN325BBClimROiRiYceaFx'
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    au_url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'.format(
    AK, SK)
    access_token = requests.post(au_url,verify=False).json()['access_token']
    headers = {'Content-Type': 'application/json'}
    api = 'https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={}'.format(access_token)
    
    if(not'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    words = pseg.cut(text)
    for w in words:
        if(w in pauseword):
            continue
        data={"text": str(w)}
        result = requests.post(api,data=json.dumps(data),headers=headers).json()
        time.sleep(0.2)
        sentiment = result['items'][0]['sentiment']
        if(sentiment==2):
            pos_count +=1
        elif(sentiment ==0):
            neg_count +=1
    cache['totalPosNum'] = pos_count
    cache['totalNegNum'] = neg_count
    return pos_count, neg_count



#总停顿词数目
def getTotalPauseWordNum(df, pauseword, cache):
    target_pattern = re.compile('|'.join(pauseword))
    text = ""
    if(not 'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    totalPauseWordNum = len(target_pattern.findall(text))
    cache['totalPauseWordNum'] = totalPauseWordNum
    return totalPauseWordNum 


#每句话时长(秒)
def getSentLen(df, cache):
    sentLen = np.array(df[df['textLength']>0]['timeLength'], dtype=np.float64)/1000
    cache['sentLen'] = sentLen
    return sentLen


#每句话字数
def getCharNum(df, cache):
    charNum = np.array([len(re.findall('\w',re.sub('_','',str(x)))) for x in df[df['textLength']>0]['text']],  dtype=np.float64)
    cache['charNum'] = charNum
    return charNum


#每句话语速
def getSentSpeed(df, cache):
    sentLen, charNum = 0.0, 0.0
    if(not ('sentLen') in cache.keys()):
        sentLen = getSentLen(df, cache)
    else:
        sentLen = cache['sentLen']
    if(not ('charNum' in cache.keys())):
        charNum = getCharNum(df, cache)
    else:
        charNum = cache['charNum']
    sentSpeed = charNum/sentLen
    cache['sentSpeed']= sentSpeed
    return sentSpeed



# New Features
def getAvgSpeed(df, cache):
    avgSpeed = 0.0
    if(not 'sentSpeed' in cache.keys()):
        sentSpeed = getSentSpeed(df, cache)
        avgSpeed = sum(sentSpeed)/sentSpeed.shape[0]
    else:
        sentSpeed = cache['sentSpeed']
        avgSpeed = sum(sentSpeed)/sentSpeed.shape[0]
    cache['avgSpeed'] = avgSpeed
    return avgSpeed

#以2秒停顿为分句，每个短句字数方差
def getCharNumVar(df, cache):
    charNumVar = 0.0
    if(not (('avgCharNum' in cache.keys() and 'charNum' in cache.keys()))):
        avgCharNum = getAvgCharPerSent(df, cache)
        charNum = getCharNum(df, cache)
        charNumVar = sum(map(lambda x: (x-avgCharNum)*(x-avgCharNum), charNum))/charNum.shape[0]
    else:
        charNumVar = sum(map(lambda x: (x-cache['avgCharPerSent'])*(x-cache['avgCharPerSent']), cache['charNum']))/cache['charNum'].shape[0]

    cache['charNumVar'] = charNumVar
    return charNumVar


#以2秒停顿为分句，每个短句的语速方差
def getSpeedVar(df, cache):
    speedChar = 0.0
    if(not ('avgSpeed' in cache.keys() and 'sentSpeed' in cache.keys())):
        avgSpeed = getAvgSpeed(df, cache)
        sentSpeed = getSentSpeed(df, cache)
        speedVar =sum(map(lambda x: (x-avgSpeed)*(x-avgSpeed), sentSpeed))/sentSpeed.shape[0]
    else:
        speedVar =sum(map(lambda x: (x-cache['avgSpeed'])*(x-cache['avgSpeed']), cache['sentSpeed']))/cache['sentSpeed'].shape[0]
    cache['speedVar'] = speedVar
    return speedVar


def getDuplicate1Abs(df, cache):
    #只计算重复的连续
    N=1
    text = ""
    totalCharNum = 0
    if(not 'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    repeat_num = 0
    two_gram_dict = []
    for i in range(len(text)):
        two_gram_dict.append([text[i:i+N]])
    for index,value in enumerate(two_gram_dict):
        if index < len(two_gram_dict)-1:
            if two_gram_dict[index] == two_gram_dict[index + 1]:
                if not str.isdigit(str(two_gram_dict[index])): #去除数字
                    repeat_num += 1
    cache['repeat_num_1'] = repeat_num
    return repeat_num

#重复一个字占所有字数百分比
# 基于张文矜代码有修改
def getDuplicate1Percent(df, cache):
    #只计算重复的连续
    totalCharNum, repeat_num_1 = 0, 0
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if(not 'repeat_num_1' in cache.keys()):
        repeat_num_1 = getDuplicate1Abs(df, cache)
    else:
        repeat_num_1 = cache['repeat_num_1']
    if(totalCharNum!=0):
        return repeat_num_1/totalCharNum
    else:
        return 0

def getDuplicate2Abs(df, cache):
    #只计算重复的连续
    N=2
    text = ""
    totalCharNum = 0
    if(not 'text' in cache.keys()):
        text = getText(df, cache)
    else:
        text = cache['text']
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    repeat_num = 0
    two_gram_dict = []
    for i in range(len(text)):
        two_gram_dict.append([text[i:i+N]])
    for index,value in enumerate(two_gram_dict):
        if index < len(two_gram_dict)-1:
            if two_gram_dict[index] == two_gram_dict[index + 1]:
                if not str.isdigit(str(two_gram_dict[index])): #去除数字
                    repeat_num += 1
    cache['repeat_num_2'] = repeat_num
    return repeat_num

#重复两个字所占百分比
#基于张文矜代码有修改
def getDuplicate2Percent(df, cache):
    #只计算重复的连续
    totalCharNum, repeat_num_2 = 0, 0
    if(not 'totalCharNum' in cache.keys()):
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if(not 'repeat_num_2' in cache.keys()):
        repeat_num_2 = getDuplicate2Abs(df, cache)
    else:
        repeat_num_2 = cache['repeat_num_2']
    if(totalCharNum!=0):
        return repeat_num_2/totalCharNum
    else:
        return 0

# Here are the new features for Sea project
def getLongSentPercent(df, cache):
    if 'charNum' not in cache.keys():
        charNum = getCharNum(df, cache)
    else:
        charNum = cache['charNum']
    long_sent_percent = np.sum(charNum > np.mean(charNum))/len(charNum)
    cache['long_sent_percent'] = long_sent_percent
    return long_sent_percent

def getEnWordPercent(df, cache):
    clean_sent = [re.findall('\w',str(x)) for x in df[df['textLength']>0]['text']]
    enWordpercent = np.sum(pd.Series([x == ['嗯'] for x in clean_sent]))/len(clean_sent)
    cache['en_word_percent'] = enWordpercent
    return enWordpercent

def getLongBlankNum(df, cache):
    inner_df = df.copy()
    # trim on rows based on whether it is null
    inner_df = inner_df[inner_df['textLength']==0]
    LongBlankCount = np.sum(inner_df['timeLength'] > 10000)
    cache['LongBlankNum'] = LongBlankCount
    return LongBlankCount

def getLongChatNum(df, cache):
    inner_df = df.copy()
    # filter out the blank sentence last shorter than 1.5 second
    inner_df = inner_df[inner_df.apply(lambda x: x['textLength'] > 0 or x['timeLength'] > 1500, axis = 1)]
    inner_df.reset_index(drop = True, inplace = True)
    inner_df['group_mark'] = inner_df['textLength']>0
    inner_df['group_mark'] = inner_df['group_mark'].apply(lambda x: 0 if x else 1)
    inner_df['group_mark'] = np.cumsum(inner_df['group_mark']) + (1 - inner_df['group_mark'])*inner_df.shape[0]
    allsent_count = inner_df.groupby('group_mark').apply(lambda x: x.shape[0])
    chatsent_count = allsent_count[np.sum(inner_df['textLength']==0):]
    LongChatCount = np.sum(chatsent_count >= 2)
    cache['LongChatNum'] = LongChatCount
    return LongChatCount

# define the helper function to find out total talk length
def getAliLenPercent(df, path_to_wav, cache):
    fileLen, AliLen = 0.0, 0.0
    if(not('fileLen' in cache.keys())):
        fileLen =  getFileLen(path_to_wav, cache)
    else:
        fileLen = cache['fileLen']
    AliLen = float(np.sum(df['timeLength']))/1000
    cache['AliLen'] = AliLen
    AliLenPercent = AliLen/fileLen
    return AliLenPercent

# this line help us load the word list files into our work space
def getWordList(path_to_words):
    with open(path_to_words, 'r', encoding='utf-8') as word_file:
        word_list = word_file.readlines()
    word_file.close()
    word_list = [re.sub('[\n|，|,| ]', '', str(x)) for x in word_list]
    return word_list
        

# start from this line, we will rewrite the basic functions and make sure the return value of these functions
# are at the sentence level
def getSentLenVector(df, cache):
    sent_len_vector = df['timeLength'].values / 1000
    cache['SentLenVector'] = sent_len_vector
    return sent_len_vector

def getCharNumVector(df, cache):
    temp = df['text'].fillna('')
    char_num_vector = np.array([len(re.findall(r'\w',str(x))) for x in temp],  dtype=np.float64)
    cache['CharNumVector'] = char_num_vector
    return char_num_vector

def getSentSpeedVector(df, cache):
    if 'SentLenVector' not in cache.keys():
        sent_len_vector = getSentLenVector(df, cache)
    else:
        sent_len_vector = cache['SentLenVector']
    if 'charNumVector' not in cache.keys():
        char_num_vector = getCharNumVector(df, cache)
    else:
        char_num_vector = cache['CharNumVector']
    # since the unit for time is ms, we need to times 1000 to the nominator
    sent_speed_vector = char_num_vector/sent_len_vector
    cache['SentSpeedVector'] = sent_speed_vector
    return sent_speed_vector

def getQuestionSentNumVector(df, cache):
    # here we still keep the same logic as total sentnum function
    temp = df['text'].fillna('')
    question_sentnum_vector = temp.apply(lambda x: x.count('？')+x.count('?')).values
    return question_sentnum_vector

# This line is used to cut the words for each sentence
def getPOSVector(df, cache):
    temp = df['text'].fillna('')
    pos = [getPOS(x, dict()) for x in temp]
    cache['POS'] = pos
    return pos

def getNounNumVector(df, cache):
    if 'POS' not in cache.keys():
        pos = getPOSVector(df, cache)
    else:
        pos = cache['POS']
    noun_num_vector = [len(x[0]) for x in pos]
    return noun_num_vector

def getVerbNumVector(df, cache):
    if 'POS' not in cache.keys():
        pos = getPOSVector(df, cache)
    else:
        pos = cache['POS']
    verb_num_vector = [len(x[1]) for x in pos]
    return verb_num_vector

def getAdjNumVector(df, cache):
    if 'POS' not in cache.keys():
        pos = getPOSVector(df, cache)
    else:
        pos = cache['POS']
    adj_num_vector = [len(x[2]) for x in pos]
    return adj_num_vector

def getAdvNumVector(df, cache):
    if 'POS' not in cache.keys():
        pos = getPOSVector(df, cache)
    else:
        pos = cache['POS']
    adv_num_vector = [len(x[3]) for x in pos]
    return adv_num_vector

def getPauseWordNumVector(df, pauseword, cache):
    temp = df['text'].fillna('')
    target_pattern = re.compile('|'.join(pauseword))
    pause_wordnum_vector = temp.apply(lambda x: len(target_pattern.findall(x))).values
    return pause_wordnum_vector


def getXNumVector(df, X_list, cache):
    temp = df['text'].fillna('')
    target_pattern = re.compile('|'.join(X_list))
    xnum_vector = temp.apply(lambda x: len(target_pattern.findall(x))).values
    return xnum_vector

# this small function is a helper function to find out two consecutive identical words
def consecutiveWordCount(text, n):
    # first step, filter out any number or character
    temp = re.sub('[^\w]|_|\d','', str(text))
    # based on the length of temp we need to go through the whole string
    consecutive_word_count = np.sum([temp[x:x+n]==temp[x+1:x+n+1] for x in range(len(temp)-1)])
    return consecutive_word_count

def getDuplicate1AbsVector(df, cache):
    temp = df['text'].fillna('')
    dup_1abs_vector = temp.apply(lambda x: consecutiveWordCount(x, 1)).values
    cache['Duplicate1AbsVector'] = dup_1abs_vector
    return dup_1abs_vector

def getDuplicate1PercentVector(df, cache):
    if 'Duplicate1AbsVector' not in cache.keys():
        dup_1abs_vector = getDuplicate1AbsVector(df, cache)
    else:
        dup_1abs_vector = cache['Duplicate1AbsVector']
    if 'CharNumVector' not in cache.keys():
        char_num_vector = getCharNumVector(df, cache)
    else:
        char_num_vector = cache['CharNumVector']
    # fill infinite with 0
    dup_1percent_vector = dup_1abs_vector/char_num_vector
    dup_1percent_vector[np.isinf(dup_1percent_vector)] = 0
    dup_1percent_vector[np.isnan(dup_1percent_vector)] = 0
    return dup_1percent_vector

def getDuplicate2AbsVector(df, cache):
    temp = df['text'].fillna('')
    dup_2abs_vector = temp.apply(lambda x: consecutiveWordCount(x, 2)).values
    cache['Duplicate2AbsVector'] = dup_2abs_vector
    return dup_2abs_vector

def getDuplicate2PercentVector(df, cache):
    if 'Duplicate2AbsVector' not in cache.keys():
        dup_2abs_vector = getDuplicate2AbsVector(df, cache)
    else:
        dup_2abs_vector = cache['Duplicate2AbsVector']
    if 'CharNumVector' not in cache.keys():
        char_num_vector = getCharNumVector(df, cache)
    else:
        char_num_vector = cache['CharNumVector']
    # fill infinite with 0
    dup_2percent_vector = dup_2abs_vector/char_num_vector
    dup_2percent_vector[np.isinf(dup_2percent_vector)] = 0
    dup_2percent_vector[np.isnan(dup_2percent_vector)] = 0
    return dup_2percent_vector

# Here we may add 8 more features corrresponding to the version 1.2's feature

def getThreeTimeLess(df, cache):
    # before we do here, we need to find only the words in a sentence
    if 'charNum' not in cache.keys():
        char_num = getCharNum(df, cache)
    else:
        char_num = cache['charNum']
    three_time_less = np.sum([x > 0 and x < 3 for x in char_num])
    cache['three_time_less'] = three_time_less
    return three_time_less

def getBetweenThreeTen(df, cache):
    if 'charNum' not in cache.keys():
        char_num = getCharNum(df, cache)
    else:
        char_num = cache['charNum']
    between_three_ten = np.sum([x >= 3 and x <= 10 for x in char_num])
    cache['between_three_ten'] = between_three_ten
    return between_three_ten

def getTenTimeMore(df, cache):
    if 'charNum' not in cache.keys():
        char_num = getCharNum(df, cache)
    else:
        char_num = cache['charNum']
    ten_time_more = np.sum([x > 10 for x in char_num])
    cache['ten_time_more'] = ten_time_more
    return ten_time_more
    
# this part is for ocean v1.3
def load_keyword_dict(path_to_keywords, cache):
    file_list = [x for x in os.listdir(path_to_keywords) if len(re.findall('.tsv|.txt', str(x)))>0]
    # load all files into work space
    dict_list = [getWordList(os.path.join(path_to_keywords, x)) for x in file_list]
    file_list = [re.sub('.tsv|.txt','',str(x)) for x in file_list]
    keyword_dict = dict(zip(file_list, dict_list))
    cache['keyword_dict'] = keyword_dict
    return keyword_dict

def convert_list_to_json(item):
    uniqueKeys=set(item)
    result=[]
    for i in uniqueKeys:
        result.append({'keyword':i,'word_count':item.count(i)})
    return result
    
def get_keyword_timestamp(df,path_to_keywords,subject,cache):
    if 'keyword_dict' not in cache:
        keyword_dict = load_keyword_dict(path_to_keywords, cache)
    else:
        keyword_dict=cache['keyword_dict']
    if subject not in keyword_dict:
        return []
    keyword_dict[subject].sort(key=len,reverse=True)
    pattern='|'.join(keyword_dict[subject])
    keywords=df.text.apply(lambda x : re.findall(pattern,str(x)))
    keywordId=keywords.apply(lambda x : False if len(x)==0 else True)
    # this line we need to be carefully since in default the sentence_id start from 1 but here we start it from 0
    if np.sum(keywordId) == 0:
        return []
    keywordTimestamp = df[keywordId].apply(lambda x: {'start_ts_ms': x['begin_time'], 'duration_ms': x['end_time'] - x['begin_time'], 'text': x['text'],
                   'keyword_list': convert_list_to_json(keywords.iloc[int(x['sentence_id']) - 1])}, axis=1).tolist()
    return keywordTimestamp

def getQuestionTimestamp(df,cache):
    if 'questionTimestamp' not in cache:
        questionInd = df['text'].apply(lambda x: True if len(re.findall(r'\?|？', str(x))) > 0 else False)
        questionTimestamp = [] if np.sum(questionInd) == 0 else df[questionInd].apply(lambda x: {'start_ts_ms': x['begin_time']}, axis = 1).tolist()
        cache['questionTimestamp']=questionTimestamp
    else:
        questionTimestamp=cache['questionTimestamp']
    return questionTimestamp


def getLongBlankTimestamp(df,cache):
    if 'longBlankTimestamp' not in cache:
        longBlankInd = df.apply(lambda x: x['textLength'] == 0 and x['timeLength'] > 10000, axis = 1)
        longBlankTimestamp = [] if np.sum(longBlankInd) == 0 else df[longBlankInd].apply(lambda x: {'start_ts_ms': x['begin_time']}, axis=1).tolist()
        cache['longBlankTimestamp']=longBlankTimestamp
    else:
        longBlankTimestamp=cache['longBlankTimestamp']
    return longBlankTimestamp


def getSubjectWordTimestamp(df,path_to_keywords,subject,cache):
    if 'subjectWordTimestamp' not in cache:
        subjectWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['subjectWordTimestamp']=subjectWordTimestamp
    else:
        subjectWordTimestamp=cache['subjectWordTimestamp']
    return subjectWordTimestamp


def getSubjectWordFirstTime(df,path_to_keywords,subject,cache):
    if 'subjectWordFirstTime' not in cache:
        subjectWordTimestamp=getSubjectWordTimestamp(df,path_to_keywords,subject,cache)
        subjectWordFirstTime=[] if len(subjectWordTimestamp)==0 else subjectWordTimestamp[0]
        cache['subjectWordFirstTime']=subjectWordFirstTime
    else:
        subjectWordFirstTime=cache['subjectWordFirstTime']
    return subjectWordFirstTime


def getSubjectWordLastTime(df,path_to_keywords,subject,cache):
    if 'subjectWordLastTime' not in cache:
        subjectWordTimestamp=getSubjectWordTimestamp(df,path_to_keywords,subject,cache)
        subjectWordLastTime=[] if len(subjectWordTimestamp)==0 else subjectWordTimestamp[-1]
        cache['subjectWordLastTime']=subjectWordLastTime
    else:
        subjectWordLastTime=cache['subjectWordLastTime']
    return subjectWordLastTime


def getSubjectWordMaxDistance(df,path_to_keywords,subject,cache):
    if 'subjectWordMaxDistance' not in cache:
        subjectWordTimestamp=getSubjectWordTimestamp(df,path_to_keywords,subject,cache)
        subjectWordTimestamp = [(x['start_ts_ms'], x['duration_ms']) for x in subjectWordTimestamp]
        subjectWordMaxDistance=0 if len(subjectWordTimestamp)<2 else np.diff(subjectWordTimestamp,axis=0)[:,0].max()
        cache['subjectWordMaxDistance']=subjectWordMaxDistance
    else:
        subjectWordMaxDistance=cache['subjectWordMaxDistance']
    return subjectWordMaxDistance


def getSubjectWordDensity(df,path_to_keywords,subject,cache):
    if 'subjectWordDensity' not in cache:
        subjectWordTimestamp=getSubjectWordTimestamp(df,path_to_keywords,subject,cache)
        if df.shape[0]==0:
            subjectWordDensity=0
        else:
            subjectWordDensity=len(subjectWordTimestamp)/np.sum(df['textLength']>0)
        cache['subjectWordDensity']=subjectWordDensity
    else:
        subjectWordDensity=cache['subjectWordDensity']
    return subjectWordDensity


def getNoteWordTimestamp(df,path_to_keywords,cache):
    if 'noteWordTimestamp' not in cache:
        subject='noteWord'
        noteWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['noteWordTimestamp']=noteWordTimestamp
    else:
        noteWordTimestamp=cache['noteWordTimestamp']
    return noteWordTimestamp


def getNoteWordMaxDistance(df,path_to_keywords,cache):
    if 'noteWordMaxDistance' not in cache:
        noteWordTimestamp=getNoteWordTimestamp(df,path_to_keywords,cache)
        noteWordTimestamp=[(x['start_ts_ms'], x['duration_ms']) for x in noteWordTimestamp]
        noteWordMaxDistance= 0 if len(noteWordTimestamp)<2 else np.diff(noteWordTimestamp,axis=0)[:,0].max()
        cache[noteWordMaxDistance]=noteWordMaxDistance
    else:
        noteWordMaxDistance=cache['noteWordMaxDistance']
    return noteWordMaxDistance


def getRedWordTimestamp(df,path_to_keywords,cache):
    if 'redWordTimestamp' not in cache:
        subject='redWord'
        redWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['redWordTimestamp']=redWordTimestamp
    else:
        redWordTimestamp=cache['redWordTimestamp']
    return redWordTimestamp


def getShortSentCount(df,cache):
    if 'shortSentCount' not in cache:
        shortSentCount=df.textLength.apply(lambda x: True if x>0 and x<3 else False).sum()
        cache['shortSentCount']=shortSentCount
    else:
        shortSentCount=cache['shortSentCount']
    return shortSentCount


def getMedianSentCount(df,cache):
    if 'medianSentCount' not in cache:
        medianSentCount=df.textLength.apply(lambda x: True if x>=3 and x<=10 else False).sum()
        cache['medianSentCount']=medianSentCount
    else:
        medianSentCount=cache['medianSentCount']
    return medianSentCount

def getLongSentCount(df,cache):
    if 'longSentCount' not in cache:
        longSentCount=(df.textLength>10).sum()
        cache['longSentCount']=longSentCount
    else:
        longSentCount=cache['longSentCount']
    return longSentCount


def getPauseWordTimestamp(df,path_to_keywords,cache):
    if 'pauseWordTimestamp' not in cache:
        subject='pauseWord'
        pauseWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['pauseWordTimestamp']=pauseWordTimestamp
    else:
        pauseWordTimestamp=cache['pauseWordTimestamp']
    return pauseWordTimestamp


def getPraiseWordTimestamp(df,path_to_keywords,cache):
    if 'praiseWordTimestamp' not in cache:
        subject='praiseWord'
        praiseWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['praiseWordTimestamp']=praiseWordTimestamp
    else:
        praiseWordTimestamp=cache['praiseWordTimestamp']
    return praiseWordTimestamp


def getGreetWordTimestamp(df,path_to_keywords,cache):
    if 'greetWordTimestamp' not in cache:
        subject='greetWord'
        greetWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['greetWordTimestamp']=greetWordTimestamp
    else:
        greetWordTimestamp=cache['greetWordTimestamp']
    return greetWordTimestamp


def getPracticeWordTimestamp(df,path_to_keywords,cache):
    if 'practiceWordTimestamp' not in cache:
        subject='practiceWord'
        practiceWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['practiceWordTimestamp']=practiceWordTimestamp
    else:
        practiceWordTimestamp=cache['practiceWordTimestamp']
    return practiceWordTimestamp


def getRepeatWordTimestamp(df, path_to_keywords,cache):
    if 'repeatWordTimestamp' not in cache:
        subject = 'repeatWord'
        repeatWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['repeatWordTimestamp']=repeatWordTimestamp
    else:
        repeatWordTimestamp = cache['repeatWordTimestamp']
    return repeatWordTimestamp


def getConcludeWordTimestamp(df, path_to_keywords, cache):
    if 'concludeWordTimestamp' not in cache:
        subject = 'concludeWord'
        concludeWordTimestamp=get_keyword_timestamp(df,path_to_keywords,subject,cache)
        cache['concludeWordTimestamp']=concludeWordTimestamp
    else:
        concludeWordTimestamp = cache['concludeWordTimestamp']
    return concludeWordTimestamp


def getPraiseSentCount(df,path_to_keywords,cache):
    if 'praiseSentCount' not in cache:
        praiseSentCount=len(getPraiseWordTimestamp(df,path_to_keywords,cache))
        cache['praiseSentCount']=praiseSentCount
    else:
        praiseSentCount=cache['praiseSentCount']
    return praiseSentCount


def getGreetSentCount(df,path_to_keywords,cache):
    if 'greetSentCount' not in cache:
        greetSentCount=len(getGreetWordTimestamp(df,path_to_keywords,cache))
        cache['greetSentCount']=greetSentCount
    else:
        greetSentCount=cache['greetSentCount']
    return greetSentCount


def getTalkTurnTimestamp(df, cache):
    turnId=df.apply(lambda x:True if x.textLength==0 and x.timeLength>1500 else False,axis=1)
    talkTurnTimestamp = [] if np.sum(turnId)==0 else df[turnId].apply(lambda x:{'start_ts_ms':x.end_time+1},axis=1).tolist()
    cache['talkTurnTimestamp'] = talkTurnTimestamp
    return talkTurnTimestamp


def getTalkTurnNum(df, cache):
    talkTurnNum=df[(df.textLength==0) & (df.timeLength>1500)].shape[0]
    cache['talkTurnNum'] = talkTurnNum
    return talkTurnNum

# this is for the truncated ali parse result
def getClassTimeLen(df, cache):
    classTimeLen = (np.max(df['end_time']) - np.min(df['begin_time']))/1000
    cache['classTimeLen'] = classTimeLen
    return classTimeLen

def getSpeedByClass(df, cache):
    if 'totalCharNum' not in cache.keys():
        totalCharNum = getTotalCharNum(df, cache)
    else:
        totalCharNum = cache['totalCharNum']
    if 'classTimeLen' not in cache.keys():
        classTimeLen = getClassTimeLen(df, cache)
    else:
        classTimeLen = cache['classTimeLen']
    speedByClass = totalCharNum/classTimeLen if classTimeLen !=0 else 0
    cache['speedByClass'] = speedByClass
    return speedByClass

def getVoiceOverClassPercent(df, cache):
    if 'classTimeLen' not in cache:
        classTimeLen = getClassTimeLen(df, cache)
    else:
        classTimeLen = cache['classTimeLen']
    if 'voiceLen' not in cache:
        voiceLen = getVoiceLen(df, cache)
    else:
        voiceLen = cache['voiceLen']
    voiceOverClassPercent = voiceLen/classTimeLen if classTimeLen !=0 else 0
    cache['voiceOverClassPercent'] = voiceOverClassPercent

def getPraiseWordCount(df, cache):
    if 'praiseWordTimestamp' not in cache:
        praiseWordTimestamp = getPraiseWordTimestamp(df,None,cache)
    else:
        praiseWordTimestamp = cache['praiseWordTimestamp']
    praiseWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in praiseWordTimestamp])
    return praiseWordCount


def getNoteWordCount(df, cache):
    if 'noteWordTimestamp' not in cache:
        noteWordTimestamp = getNoteWordTimestamp(df,None,cache)
    else:
        noteWordTimestamp = cache['noteWordTimestamp']
    noteWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in noteWordTimestamp])
    return noteWordCount


def getRedWordCount(df, cache):
    if 'redWordTimestamp' not in cache:
        redWordTimestamp = getRedWordTimestamp(df,None,cache)
    else:
        redWordTimestamp = cache['redWordTimestamp']
    redWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in redWordTimestamp])
    return redWordCount


def getPracticeWordCount(df, cache):
    if 'practiceWordTimeStamp' not in cache:
        practiceWordTimestamp = getPracticeWordTimestamp(df,None,cache)
    else:
        practiceWordTimestamp = cache['PracticeWordTimestamp']
    practiceWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in practiceWordTimestamp])
    return practiceWordCount

def getGreetWordCount(df, cache):
    if 'greetWordTimestamp' not in cache:
        greetWordTimestamp = getGreetWordTimestamp(df,None,cache)
    else:
        greetWordTimestamp = cache['greetWordTimestamp']
    # greetWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in greetWordTimestamp if x['start_ts_ms'] < 36000])
    greetWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in greetWordTimestamp])
    return greetWordCount

def getRepeatWordCount(df, cache):
    if 'repeatWordTimestamp' not in cache:
        repeatWordTimestamp = getRepeatWordTimestamp(df,None,cache)
    else:
        repeatWordTimestamp = cache['repeatWordTimestamp']
    # repeatWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in repeatWordTimestamp if x['start_ts_ms'] > np.max(df['end_time'])-54000])
    repeatWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in repeatWordTimestamp])
    return repeatWordCount

def getConcludeWordCount(df, cache):
    if 'concludeWordTimestamp' not in cache:
        concludeWordTimestamp = getConcludeWordTimestamp(df,None,cache)
    else:
        concludeWordTimestamp = cache['concludeWordTimestamp']
    # concludeWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in concludeWordTimestamp if x['start_ts_ms'] > np.max(df['end_time'])-54000])
    concludeWordCount = np.sum([np.sum([y['word_count'] for y in x['keyword_list']]) for x in concludeWordTimestamp])
    return concludeWordCount

# this is for godeye 1.5, timeseries feature extraction

def getTimestampStats(keywordTimestamp):
    # we transform them into second unit
    timestamps = [float(x['start_ts_ms'])/1000 for x in keywordTimestamp]
    timestamps = sorted(timestamps)
    result = [len(timestamps), np.mean(timestamps), np.std(timestamps), np.var(timestamps), skew(timestamps), kurtosis(timestamps), timestamps[int(len(timestamps)*0.8)]-timestamps[int(len(timestamps)*0.2)], np.percentile(timestamps, 25), np.percentile(timestamps, 50), np.percentile(timestamps, 75)] if len(timestamps) !=0 else [0]*10
    columns = ['Count', 'Mean', 'Std', 'Variance', 'Skew', 'Kurtosis', 'Range', '25Per','50Per','75Per']
    return result, columns

def getSubjectWordTimestampStats(df, cache):
    if 'subjectWordTimestamp' not in cache:
        keyWordTimestamp = getSubjectWordTimestamp(df, None, cache['subject'], cache)
    else:
        keyWordTimestamp = cache['subjectWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getQuestionTimestampStats(df, cache):
    if 'questionTimestamp' not in cache:
        keyWordTimestamp = getQuestionTimestamp(df, cache)
    else:
        keyWordTimestamp = cache['questionTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getNoteWordTimestampStats(df, cache):
    if 'noteWordTimestamp' not in cache:
        keyWordTimestamp = getNoteWordTimestamp(df, None, cache)
    else:
        keyWordTimestamp = cache['noteWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getRedWordTimestampStats(df, cache):
    if 'redWordTimestamp' not in cache:
        keyWordTimestamp = getRedWordTimestamp(df, None, cache)
    else:
        keyWordTimestamp = cache['redWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getPauseWordTimestampStats(df, cache):
    if 'pauseWordTimestamp' not in cache:
        keyWordTimestamp = getPauseWordTimestamp(df, None, cache)
    else:
        keyWordTimestamp = cache['pauseWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getPraiseWordTimestampStats(df, cache):
    if 'praiseWordTimestamp' not in cache:
        keyWordTimestamp = getPraiseWordTimestamp(df, None, cache)
    else:
        keyWordTimestamp = cache['praiseWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getGreetWordTimestampStats(df, cache):
    if 'greetWordTimestamp' not in cache:
        keyWordTimestamp = getGreetWordTimestamp(df, None, cache)
    else:
        keyWordTimestamp = cache['greetWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getPracticeWordTimestampStats(df, cache):
    if 'practiceWordTimeStamp' not in cache:
        keyWordTimestamp = getPracticeWordTimestamp(df, None, cache)
    else:
        keyWordTimestamp = cache['practiceWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getRepeatWordTimestampStats(df, cache):
    if 'repeatWordTimeStamp' not in cache:
        keyWordTimestamp = getRepeatWordTimestamp(df, None, cache)
    else:
        keyWordTimestamp = cache['repeatWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns

def getConcludeWordTimestampStats(df, cache):
    if 'concludeWordTimestamp' not in cache:
        keyWordTimestamp = getConcludeWordTimestamp(df, None, cache)
    else:
        keyWordTimestamp = cache['concludeWordTimestamp']
    stats_result, stats_columns = getTimestampStats(keyWordTimestamp)
    return stats_result, stats_columns
