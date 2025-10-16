import json
from datasets import Dataset, DatasetDict

annotations_path = "./dataset/python/train.jsonl"

# JSONLファイルを1行ずつ読み込む
annotations = []
with open(annotations_path, 'r') as f:
    for line in f:
        annotations.append(json.loads(line))

# annotationsを確認

# 各行が辞書であることを確認した上で、キーごとにカラム名を作成
dataset = Dataset.from_dict({
    'index': [example['index'] for example in annotations],
    'code': [example['code'] for example in annotations],
    'contrast': [example['contrast'] for example in annotations],
    'label': [example['label'] for example in annotations]
})

# DatasetDict に変換し、'train' split を追加
dataset_dict = DatasetDict({
    'train': dataset
})

# データセットを保存
dataset_dict.save_to_disk("./converted_dataset")