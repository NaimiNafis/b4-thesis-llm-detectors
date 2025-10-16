# 必要なライブラリのインポート
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import logging
import random
import numpy as np
from torch.utils.data import Dataset
from transformers import TrainerCallback
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # または ":4096:8"
# シードを固定する関数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# worker_init_fnの定義
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# カスタムデータセットクラス
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_text = f"Code: {example['code']} "
        label = example['label']

        # トークナイズ
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        return inputs

# シードを固定
set_seed(42)

# ハイパーパラメータのグリッド
grid_num_train_epochs = [2, 3]
grid_learning_rates = [1e-5, 5e-5, 1e-6]
batch_sizes = [16, 32]
grid_gradient_accumulation_steps = [8, 16, 32, 2]


# データセットのロード
ds = load_from_disk("./converted_dataset/train")

# 結果を保存するファイル
results_file = "./result_grid.txt"
with open(results_file, "w") as outfile:
    outfile.write("")

# モデルチェックポイント
ckpt = "microsoft/graphcodebert-base"
# ckpt = "bert-base-cased"

# トレーニングループ
for train_epoch in grid_num_train_epochs:
    for rate in grid_learning_rates:
        for accumulation_step in grid_gradient_accumulation_steps:
            for batch_size in batch_sizes:

                # モデルとトークナイザーのロード
                tokenizer = AutoTokenizer.from_pretrained(ckpt)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForSequenceClassification.from_pretrained(
                    ckpt,
                    num_labels=2,
                    torch_dtype=torch.float32,
                    device_map="auto",
                )
                for param in model.parameters():
                    param.requires_grad = True

                # カスタムデータセットを作成
                train_dataset = CustomDataset(ds, tokenizer)

                # トレーニング設定
                args = TrainingArguments(
                    num_train_epochs=train_epoch,
                    remove_unused_columns=False,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation_step,
                    warmup_steps=500,
                    learning_rate=rate,
                    weight_decay=0.01,
                    logging_steps=100,
                    save_strategy="epoch",
                    save_total_limit=1,
                    save_steps=5000, 
                    optim="adamw_torch",
                    output_dir="./fine_tuned_model",
                    dataloader_pin_memory=True,
                )

                # ロギング設定
                logging.basicConfig(
                    filename="training_progress.log",
                    filemode="w",
                    format="%(asctime)s - %(message)s",
                    level=logging.INFO,
                )

                class LoggingCallback(TrainerCallback):
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if logs:
                            logging.info(f"Epoch {state.epoch}: {logs}")

                # トレーナーの作成
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    data_collator=None,  # カスタムデータセットで既にトークナイズ済みのため不要
                    callbacks=[LoggingCallback()],
                )

                # トレーニングの実行
                trainer.train()

                # モデルとトークナイザーを保存
                trainer.save_model("./fine_tuned_model")
                tokenizer.save_pretrained("./fine_tuned_model")

                # 推論関数
                def predict(input_code):
                    inputs = tokenizer(
                        f"Code: {input_code}",
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                    )
                    inputs = {key: value.to(model.device) for key, value in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits
                    return torch.argmax(logits, dim=-1).item()

                # テストデータの評価
                test_file = "./valid.jsonl"
                true_labels = []
                pred_labels = []
                with open(test_file, "r") as infile:
                    for line in infile:
                        data = json.loads(line)
                        index = data["index"]
                        code = data["code"]
                        true_label = data["label"]

                        pred_label = predict(code)
                        true_labels.append(true_label)
                        pred_labels.append(pred_label)

                # 評価指標の計算
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, pred_labels, average=None, labels=[1, 0]
                )
                accuracy = accuracy_score(true_labels, pred_labels)

                # 結果の保存
                print(f"Parameters: Epoch={train_epoch}, LR={rate}, Accumulation={accumulation_step}, Batch={batch_size}")
                print(f"LLM Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1[0]:.4f}")
                print(f"Human Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1[1]:.4f}")
                print(f"Accuracy: {accuracy:.4f}")
                with open(results_file, "a") as outfile:
                    outfile.write(f"Parameters: Epoch={train_epoch}, LR={rate}, Accumulation={accumulation_step}, Batch={batch_size}\n")
                    outfile.write(f"LLM Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1[0]:.4f}\n")
                    outfile.write(f"Human Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1[1]:.4f}\n")
                    outfile.write(f"Accuracy: {accuracy:.4f}\n\n")