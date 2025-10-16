import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch

torch.manual_seed(42) 
model_path = "./fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

# モデルのロード
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(input_code):
    """
    入力コードに基づいてモデルの予測を行う。
    LLMによるコードは1、人間によるコードは0を返す。
    """
    # 入力コードをトークナイズ
  # 入力コードをトークナイズ
    inputs = tokenizer(f"Code: {input_code}", return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # モデルに入力して出力のlogitsを取得
    with torch.no_grad():
        outputs = model(**inputs)

    # 出力のlogits（最後の層の出力）から予測を計算
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()  # 最小のlogitsを選択

    return predicted_class  # 0または1を返す

# ----- テストデータのロード -----
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

        # 推論
        pred_label = predict(code)
        true_labels.append(true_label)
        pred_labels.append(pred_label)

        # 結果の書き込み
        result = {
            "index": index,
            "true_label": true_label,
            "predicted_label": pred_label
        }
        outfile.write(json.dumps(result) + "\n")

# ----- 評価指標の計算 -----
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[1, 0])
accuracy = accuracy_score(true_labels, pred_labels)

# 評価結果の表示
print("Evaluation Results:")
print(f"LLM Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1[0]:.4f}")
print(f"Human Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1[1]:.4f}")
print(f"Accuracy: {accuracy:.4f}")

with open(results_file, "a") as outfile:
    outfile.write(f"LLM Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1[0]:.4f}\n")
    outfile.write(f"Human Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1[1]:.4f}\n")
    outfile.write(f"Accuracy: {accuracy:.4f}\n\n")
