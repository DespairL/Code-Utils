# Data Preparation for Post-Training

This document outlines the data preparation process for VERL post-training jobs, based on the official VERL documentation <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>.

## Overview

Before initiating post-training jobs, data must be prepared and stored in **parquet format** <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>. VERL provides preprocessing scripts for several datasets including GSM8K, MATH, HelloSwag, and Full_hh_rlhf <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>.

## Data Preprocessing Architecture

The data preprocessing pipeline consists of two main components <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>:

### 1. Common Processing Framework
- Loads datasets from HuggingFace's datasets package <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- Applies custom `make_map_fn` preprocessing function <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- Stores processed data in parquet format <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>

### 2. Custom Implementation Requirements
Users must implement the `make_map_fn()` function and `extract_solution()` function to support different datasets or tasks <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>.

## Required Data Schema

Each processed data entry must contain the following five fields <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>:

### 1. `data_source`
- **Purpose**: Dataset identifier for indexing corresponding reward functions in RewardModule <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- **Type**: String

### 2. `prompt`
- **Purpose**: Input formatted according to HuggingFace chat_template <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- **Processing**: Tokenizer in RLHFDataset applies chat template and tokenization <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- **Format**: List of role-content dictionaries

### 3. `ability`
- **Purpose**: Task category definition <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- **Type**: String (e.g., "math")

### 4. `reward_model`
- **Purpose**: Ground truth for evaluation <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- **Components**:
  - `style`: Reward computation method
  - `ground_truth`: Extracted solution via `extract_solution()` function <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- **Note**: Reward function implementation must align with extracted ground truth <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>

### 5. `extra_info`
- **Purpose**: Additional prompt metadata <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
- **Status**: Currently unused <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>

## Implementation Example: GSM8K Dataset

### Solution Extraction Function
```python
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution
```

### Data Processing Function
```python
def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop('question')
        question = question + ' ' + instruction_following
        
        answer = example.pop('answer')
        solution = extract_solution(answer)
        
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn
```

## Key Considerations

1. **Format Consistency**: All prompts must follow HuggingFace chat_template format <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
2. **Reward Alignment**: Reward function implementation must match ground truth extraction logic <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
3. **Storage Format**: Final output must be in parquet format for efficient processing <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>
4. **Custom Implementation**: Each dataset requires tailored `make_map_fn()` and `extract_solution()` functions <mcreference link="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"  ></mcreference>

This standardized approach ensures compatibility with VERL's post-training pipeline while maintaining flexibility for diverse dataset requirements.
        