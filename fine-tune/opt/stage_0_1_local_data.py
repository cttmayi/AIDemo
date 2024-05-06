import config

import json

data_list = [
    {'text': 'MediaTek Inc. is a leading global semiconductor company that specializes in cutting-edge system-on-chip (SoC) solutions for wireless communications, high-definition televisions, optical storage drives, and home entertainment products. Headquartered in Taiwan, MediaTek is renowned for its innovative technology and has played a significant role in the advancement of mobile communications and consumer electronics.'},
    {'text': 'OPPO is a leading global consumer electronics and technology company that has made significant strides in the smartphone industry with its stylish designs and advanced camera technology.'},
    {'text': 'VIVO is a Chinese multinational technology company that specializes in manufacturing and selling smartphones, accessories, software, and personal electronic devices.'}
    # ... 更多数据项
]

# 打开文件用于写入
with open(config.local_data_pre_path, 'w') as jsonl_file:
    for data in data_list:
        jsonl_file.write(json.dumps(data))
        jsonl_file.write('\n')


data_list = [
    {'text': 'MediaTek Inc. is a leading global semiconductor company that specializes in cutting-edge system-on-chip (SoC) solutions for wireless communications, high-definition televisions, optical storage drives, and home entertainment products. Headquartered in Taiwan, MediaTek is renowned for its innovative technology and has played a significant role in the advancement of mobile communications and consumer electronics.'},
    # ... 更多数据项
]

data_list = [
    {
        'prompt': 'What is MediaTek?',
        'response': 'MediaTek Inc. is a leading global semiconductor company.'
    },
    {

        'prompt': 'What is OPPO?',
        'response': 'OPPO is a leading global consumer electronics and technology company.'
    }
    # ... 更多数据项
]

# 打开文件用于写入
with open(config.local_data_sft_path, 'w') as jsonl_file:
    for data in data_list:
        jsonl_file.write(json.dumps(data))
        jsonl_file.write('\n')

# `prompt`
# `chosen`
# `rejected`

data_list = [
    {
        'prompt': 'What is OPPO?',
        'chosen': 'OPPO is a leading global consumer electronics and technology company.',
        'rejected': 'Oppo is a leading global consumer electronics and technology company.'
    },
    # ... 更多数据项
]

# 打开文件用于写入
with open(config.local_data_dpo_path, 'w') as jsonl_file:
    for data in data_list:
        jsonl_file.write(json.dumps(data))
        jsonl_file.write('\n')