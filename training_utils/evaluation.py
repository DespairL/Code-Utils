# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
import json

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local

@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)

@ray.remote
def process_item_with_custom_reward(task_name, index, data_source, response_lst, reward_data):
    """在Ray worker中加载并使用自定义奖励函数"""
    import sys
    sys.path.append('/home/nfs06/chenyq/Code-Utils')
    from reward_utils.custom_reward import get_reward_function
    # Get the corresponding reward function
    reward_fn = get_reward_function(task_name)
    
    ground_truth = reward_data["ground_truth"]
    ret_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    score = ret_lst[0]['reward']
    prediction = ret_lst[0]['prediction']
    # return data_source, np.mean(score_lst)

    return {
        'index': index,
        'data_source': data_source,
        'scores': score,
        'mean_score': float(np.mean(score)),
        'ground_truth': ground_truth,
        'responses': response_lst[0],
        'prediction': prediction,
    }

@hydra.main(config_path="config", config_name="custom_eval", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # Use custom reward function
    data_source_reward = defaultdict(list)
    all_results = []  # 存储所有结果
    task_name = config.custom_reward_function.get('task_name', 'gsm8k')
    print(f"Using custom reward function for task: {task_name}")
    
    # Create remote tasks with custom reward function
    remote_tasks = [
        process_item_with_custom_reward.remote(task_name, i, data_sources[i], 
            responses[i], reward_model_data[i]) 
        for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                result = ray.get(result_id)
                data_source = result['data_source']
                score = result['mean_score']
                data_source_reward[data_source].append(score)
                all_results.append(result)
                pbar.update(1)

    # 按索引排序结果
    all_results.sort(key=lambda x: x['index'])
    
    # 准备保存的数据
    save_data = []
    for result in all_results:
        save_data.append({
            'index': result['index'],
            'data_source': result['data_source'],
            'responses': result['responses'],
            'ground_truth': result['ground_truth'],
            'scores': result['scores'],
            'mean_score': result['mean_score']
        })

    input_file_path = config.data.path
    if input_file_path.endswith('.parquet'):
        output_file_path = input_file_path.replace('.parquet', '.json')
    else:
        output_file_path = input_file_path + '.json'
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"Rewards saved to: {output_file_path}")

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    print(metric_dict)


if __name__ == "__main__":
    main()
