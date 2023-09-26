import sys
import os

basePath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(basePath, 'tools'))

from tools.basic_module import *
import tools.util_tools as util_tools
from tools.classEvalution import get_class_evaluation

def check_input(student_json=None, teacher_json=None, student_start_at=0, teacher_start_at=0, task_id=''):
    error_code = default_error_code
    error_message = default_error_message
    result = {}
    error_code_student, error_message_student = util_tools.check_input_text(
        student_json)

    if error_code_student != 0:
        error_code = error_code_student
        error_message = error_message_student
        
    if not teacher_json is None and error_code==0:
        error_code_teacher, error_message_teacher = util_tools.check_input_text(
            teacher_json)
        if error_code_teacher != 0 and not teacher_json is None:
            error_code = error_code_teacher
            error_message = error_message_teacher
    return error_code, error_message, result


def evaluation(student_json, teacher_json, student_start_at=0, teacher_start_at=0, subject=12, task_id=''):
    logger.info('Start - task_id:{}'.format(task_id))
    error_code, error_message,result = check_input(
        student_json, teacher_json, student_start_at, teacher_start_at, task_id)
    if error_code == 0:
        jsonStr = util_tools.change_input_2_godeye(student_json, teacher_json, student_start_at, teacher_start_at,
                                                   subject, task_id)
        error_code, error_message, result = util_tools.check_jsonStr(jsonStr,teacher_json)
        if error_code == 0:
            error_code, error_message, result = get_class_evaluation(jsonStr, task_id)
    logger.info('Finish - task_id:{},error_code:{},error_message:{}'.format(task_id,error_code,error_message))
    return {'error_code': error_code, 'error_message': error_message, 'result': result}

if __name__ == "__main__":
    text_json = {
    "text_teacher": [
        {
            "text": "这句话呢，其实都是告诉你游戏规则，他就看你能不能看到他这个给你的规定了。",
            "begin_time": 1326750,
            "end_time": 1332165
        },
        {
            "text": "或者说你骂人一个游戏，它上面会有一个游戏的一个，这个攻略对不对？",
            "begin_time": 1334200,
            "end_time": 1339555
        },
        {
            "text": "那你是不是要看到的这个游戏规则？女足知道了你这游戏谈话。",
            "begin_time": 1339540,
            "end_time": 1344105
        },
        {
            "text": "我什么时候该拐弯了或者什么？这种游戏是不是？",
            "begin_time": 1353120,
            "end_time": 1355985
        },
        {
            "text": "规则是不是那就是你看这句话呢76三呢，是游戏给你的一个攻略，告诉你这个游戏怎么玩",
            "begin_time": 1359510,
            "end_time": 1366485
        },
        {
            "text": "cc相同，是吧，那你看啊，这我就开始玩游戏的事嘛，对不对，那就按照他的这个说法来吧。",
            "begin_time": 1534130,
            "end_time": 1541365
        }
    ],
    "text_student": [
        {
            "text": "这句话呢，其实都是告诉你游戏规则，他就看你能不能看到他这个给你的规定了。",
            "begin_time": 1326750,
            "end_time": 1332165
        },
        {
            "text": "或者说你骂人一个游戏，它上面会有一个游戏的一个，这个攻略对不对？",
            "begin_time": 1334200,
            "end_time": 1339555
        },
        {
            "text": "那你是不是要看到的这个游戏规则？女足知道了你这游戏谈话。",
            "begin_time": 1339540,
            "end_time": 1344105
        },
        {
            "text": "我什么时候该拐弯了或者什么？这种游戏是不是？",
            "begin_time": 1353120,
            "end_time": 1355985
        },
        {
            "text": "规则是不是那就是你看这句话呢76三呢，是游戏给你的一个攻略，告诉你这个游戏怎么玩",
            "begin_time": 1359510,
            "end_time": 1366485
        },
        {
            "text": "cc相同，是吧，那你看啊，这我就开始玩游戏的事嘛，对不对，那就按照他的这个说法来吧。",
            "begin_time": 1534130,
            "end_time": 1541365
        }
    ],
    "teacher_start_at_ms": 20,
    "student_start_at_ms": 10,
    "subject": "other"
}
print(check_input(text_json["text_teacher"],text_json["text_student"]))
print(evaluation(text_json["text_teacher"],text_json["text_student"]))