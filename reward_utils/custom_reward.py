# 创建一个独立的奖励函数模块
import numpy as np
from verl.utils.reward_score import gsm8k

# 参数需要是 data_source, solution_str, ground_truth, extra_info=None
def compute_gsm8k_score(data_source, solution_str, ground_truth, extra_info=None):
    """GSM8K 奖励函数，处理各种数据类型"""
    return gsm8k.compute_score(solution_str, ground_truth, )

# 奖励函数注册表
REWARD_FUNCTIONS = {
    'gsm8k': compute_gsm8k_score,
    # 可以添加更多自定义函数
}

def get_reward_function(task_name):
    """根据任务名称获取对应的奖励函数"""
    return REWARD_FUNCTIONS.get(task_name, compute_gsm8k_score)