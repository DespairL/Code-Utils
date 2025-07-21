import numpy as np
from verl.utils.reward_score import gsm8k

def compute_gsm8k_score(data_source, solution_str, ground_truth, extra_info=None):
    """GSM8K 奖励函数，处理各种数据类型"""
    answer = gsm8k.extract_solution(solution_str=solution_str, method='flexible')
    return {
        'reward': gsm8k.compute_score(solution_str, ground_truth, method='flexible'),
        'prediction': answer
    }

REWARD_FUNCTIONS = {
    'gsm8k': compute_gsm8k_score,
}

def get_reward_function(task_name):
    """根据任务名称获取对应的奖励函数"""
    return REWARD_FUNCTIONS.get(task_name, compute_gsm8k_score)