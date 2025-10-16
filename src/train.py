# 必要なライブラリのインポート
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import logging
from transformers import TrainerCallback
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

torch.manual_seed(42)
removed_file = "./remove_log.txt"

# データセットの読み込み
ds = load_from_disk("./converted_dataset/train")

# GraphCodeBERTの事前学習済みチェックポイント
ckpt = "microsoft/graphcodebert-base"

# トークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(ckpt)

# モデルのロード（全パラメータを学習対象に）
model = AutoModelForSequenceClassification.from_pretrained(
    ckpt,
    num_labels=2,  # LLM vs 人間の2クラス分類
    torch_dtype=torch.float32,
    device_map="auto"
    
)

# 全パラメータを学習対象に設定
for param in model.parameters():
    param.requires_grad = True

count = 0  # グローバル変数として定義
count_line = 0

# データの前処理関数
def process(examples):
    global count_line  # グローバル変数を使用
    global count

    inputs = [f"Code: {example['code']} Contrast: {example['contrast']}" for example in examples]  # `example`は辞書
    labels = [example['label'] for example in examples]  # 'label'が辞書のキーである場合

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids("<|endoftext|>")

    filtered_inputs = []
    filtered_labels = []
    count_line += 1

    for input_text, label in zip(inputs, labels):
        # トークナイズ
        tokens = tokenizer(input_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        eos_count = (tokens['input_ids'] == eos_token_id).sum().item()
        # EOSトークンが2つ未満の場合のみ追加
        if eos_count < 2:
            filtered_inputs.append(input_text)
            filtered_labels.append(label)
        if eos_count == 2:
            count=count+1

            print("\nline=",count_line)
            with open(removed_file, "a") as outfile:
                 outfile.write(f"line=: {count_line},\n")

    # トークナイズとTensor化
    model_inputs = tokenizer(
        filtered_inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )
    # ラベルをTensorに変換
    model_inputs["labels"] = torch.tensor(filtered_labels, dtype=torch.long)
    return model_inputs

# トレーニング設定
args = TrainingArguments(
    num_train_epochs=3,
    remove_unused_columns=False,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    learning_rate=1e-05,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="epoch",
    optim="adamw_torch",
    output_dir="./fine_tuned_graphcodebert",
    dataloader_pin_memory=True,

)

# ロギング設定
logging.basicConfig(
    filename="training_progress.log",
    filemode="w",
    format="%(asctime)s - %(message)s",
    level=logging.INFO
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# トレーニング中にロギングするコールバック
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info(f"Epoch {state.epoch}: {logs}")

# トレーナーの作成
trainer = Trainer(
    model=model,
    train_dataset=ds,
    data_collator=lambda data: process(data),
    args=args,
    callbacks=[LoggingCallback()]
)
# トレーニングの実行
trainer.train()

print("total of removed code :", count)
with open(removed_file, "a") as outfile:
    outfile.write(f"total of removed code : {count},\n")

# モデルとトークナイザーを保存
trainer.save_model("./fine_tuned_graphcodebert")
model.save_pretrained("./fine_tuned_graphcodebert")
tokenizer.save_pretrained("./fine_tuned_graphcodebert")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(input_code):
    """
    入力コードに基づいてモデルの予測を行う。
    LLMによるコードは1、人間によるコードは0を返す。
    """
    inputs = tokenizer(f"Code: {input_code}", return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# テストデータのロード
test_file = "./dataset/python/test.jsonl"
results_file = "./result.txt"

true_labels = []
pred_labels = []

with open(test_file, "r") as infile, open(results_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        index = data["index"]
        code = data["code"]
        true_label = data["label"]

        pred_label = predict(code)
        true_labels.append(true_label)
        pred_labels.append(pred_label)

        result = {
            "index": index,
            "true_label": true_label,
            "predicted_label": pred_label
        }
        outfile.write(json.dumps(result) + "\n")

# 評価指標の計算
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[1, 0])
accuracy = accuracy_score(true_labels, pred_labels)

print("Evaluation Results:")
print(f"LLM Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1[0]:.4f}")
print(f"Human Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1[1]:.4f}")
print(f"Accuracy: {accuracy:.4f}")

with open(results_file, "a") as outfile:
    outfile.write(f"LLM Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1[0]:.4f}\n")
    outfile.write(f"Human Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1[1]:.4f}\n")
    outfile.write(f"Accuracy: {accuracy:.4f}\n\n")