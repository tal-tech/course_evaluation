
import os
import sys
import pickle
import pandas as pd
from basic_module import *


base_path = os.path.dirname(os.path.realpath(__file__))
# load data and file
sys.path.append(os.path.join(base_path, '../columbus'))

from columbusJson import columbusJsonSDK

# load model
class_evaluate_model_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '../../data/base/model/class_evaluate.pkl')
with open(class_evaluate_model_path, 'rb') as model_file:
    class_evaluate_model = pickle.load(model_file)


def get_class_evaluation(jsonStr, task_id=''):
    error_code = default_error_code
    error_message = default_error_message
    result = {}
    logging.info('task_id: {},start class evaluation'.format(task_id))
    timer = ticktock()
    logger.info('task_id: {}'.format(task_id))
    # we make a judgement on student text only since model only need student features, later we may need to make some change
    if len(jsonStr['student']['text']) == 0 or len(jsonStr['teacher']['text']) == 0:
        cost_time = timer.timeDist()
        logging.error(
            'task_id: {} - teacher or student file is empty, time cost is {} ms!'.format(task_id, cost_time))
        error_message = 'teacher or student file is empty'
        return text_empty, error_message, result

    # since here we only use the student features, thus we only run the columbus once
    studentFeature = columbusJsonSDK(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../config/ocean_student_json.ini'), jsonStr,task_id
    )
    if studentFeature.shape[0] == 0:
        cost_time = timer.timeDist()
        logging.error(
            'task_id: {} - columbus feature process on student file failed, time cost is {} ms!'.format(task_id, cost_time))
        error_message = 'columbus feature process on student file failed'
        return columbus_error, error_message, result
    teacherFeature = columbusJsonSDK(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../config/ocean_teacher_json.ini'), jsonStr,task_id
    )
    if teacherFeature.shape[0] == 0:
        cost_time = timer.timeDist()
        logging.error(
            'task_id: {} - columbus feature process on teacher file failed, time cost is {} ms!'.format(task_id, cost_time))
        return columbus_error, error_message, result
    studentFeature.columns = ['s_' + x if x != 'classID' else x for x in studentFeature.columns.tolist()]
    teacherFeature.columns = ['t_' + x if x != 'classID' else x for x in teacherFeature.columns.tolist()]
    logging.info(
        'task_id: {},studentFeature shape is {},teacherFeature shape is {}'.format(task_id, studentFeature.shape,teacherFeature.shape))

    classFeature = pd.merge(teacherFeature, studentFeature)
    try:
        features = class_evaluate_model['feature']
        x_pred = classFeature[features]
        if np.sum(np.abs(x_pred.values)) == 0:
            cost_time = timer.timeDist()
            logging.error(
                'task_id: {} - columbus features are all zero, time cost is {} ms!'.format(task_id, cost_time))
            error_message = 'columbus features are all zero'
            return columbus_error, error_message, result
        model = class_evaluate_model['model']
        prob = model.predict_proba(x_pred)[0][1]
        score = int(100 * prob)
        level = 2 if score > 50 else 1 if score > 20 else 0
    except:
        cost_time = timer.timeDist()
        logging.error('task_id: {} - Model prediction error, time cost is {} ms, detail is {}'.format(
            task_id, cost_time, traceback.format_exc()))
        error_message = 'Model prediction error'
        return class_evaluation_error, error_message, result
    cost_time = timer.timeDist()
    result = {'class_score': score, 'class_level': level}
    logging.info(
        'task_id: {} - class prediction succeed, time cost is {} ms!'.format(task_id, cost_time))
    return error_code,error_message, result
