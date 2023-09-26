from basic_module import *
from classEvalution import get_class_evaluation

def get_sentence_list_duration_s(df, task_id=''):
    duration = (np.max(df['end_time']) - np.min(df['begin_time'])) / 1000
    return duration


def get_speak_speed_s(df, task_id=''):
    '''语速'''
    duration = get_sentence_list_duration_s(df)
    char_num = df.textLength.sum()
    speak_speed = char_num / duration
    return speak_speed


def get_sentence_num(df, task_id=''):
    '''句数'''
    return df.shape[0]


def get_sentence_average_length(df, task_id=''):
    '''平均句长'''
    return df.textLength.mean()


def get_pause_words_num(df, task_id=''):
    '''停顿词次数'''
    pause_words_num = df['text'].apply(lambda x: len(
        re_pauseword.findall(str(x))) > 0).sum()
    return pause_words_num


def get_question_num(df, task_id=''):
    '''问句数量'''
    question_num = df['text'].apply(lambda x: True if len(
        re_question.findall(str(x))) > 0 else False).sum()
    return question_num


def get_short_sentence_num(df, task_id=''):
    '''短句数量<3'''
    short_sentnce_num = df.textLength.apply(
        lambda x: True if x > 0 and x < 3 else False).sum()
    return short_sentnce_num


def get_median_sentence_num(df, task_id=''):
    '''中句数量>=3 and <=10'''
    median_sentence_num = df.textLength.apply(
        lambda x: True if x >= 3 and x <= 10 else False).sum()
    return median_sentence_num


def get_long_sentence_num(df, task_id=''):
    '''长句数量>10'''
    long_sentence_num = df.textLength.apply(
        lambda x: True if x > 10 else False).sum()
    return long_sentence_num


def get_interaction_score(df, task_id=''):
    t = get_sentence_list_duration_s(df)
    k = get_sentence_num(df)
    k1 = get_short_sentence_num(df)
    k2 = get_median_sentence_num(df)
    k3 = get_long_sentence_num(df)
    m = get_question_num(df)
    v_speak_num = 1 if k / t > 0.063 else 0
    v_duration = 1 if (k1 + k2 * 3 + k3 * 6) / t > 0.27 else 0
    v_question = 1 if m / t * 100 > 0.896 else 0
    if v_speak_num and v_duration and v_question:
        level = 0
    if v_speak_num and v_duration and not v_question:
        level = 1
    if v_speak_num and not v_duration:
        level = 2
    if not v_speak_num:
        level = 3
    return level

