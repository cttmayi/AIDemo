import csv
import json
from src.utils.log_parser import auto_parser

def csv_read(data_path):
    with open(data_path, mode='r', encoding='utf-8') as file:
        # 创建一个csv读取器
        csv_reader = csv.reader(file)
        
        rows = []
        for l, row in enumerate(csv_reader):
            if l == 0:
                headers = row
                continue
            row_dict = dict(zip(headers, row))
            rows.append(row_dict)
        return rows
    return None


def csv_write(data_path, data, header=None):
    with open(data_path, mode='w', encoding='utf-8') as file:
        csv_writer = csv.writer(file)

        if header is None:
            headers = data[0].keys()
        csv_writer.writerow(headers)

        for row in data:
            row = [row[key] for key in headers]
            csv_writer.writerow(row)


def jsonl_read(data_path):
    with open(data_path, mode='r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        return data
    return None


def jsonl_write(data_path, data):
    with open(data_path, mode='w', encoding='utf-8') as file:
        for row in data:
            file.write(json.dumps(row) + '\n')


def log_read(data_path):
    with open(data_path, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        data, type = auto_parser(lines)
        return data, type
    return None, None