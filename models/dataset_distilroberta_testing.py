import os
import torch
import pandas as pd
from tqdm import tqdm  # For progress bar
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import logging  # For logging evaluation metrics

# =====================
# Configure Paths and Logging
# =====================
base_logs_dir = os.path.abspath("logs")  # Ensure absolute path for logs folder
os.makedirs(base_logs_dir, exist_ok=True)  # Create logs folder if it doesn't exist
output_false_file = "logs/distilroberta_false_payloads.csv"
log_file = "logs/distilroberta_evaluation.log"

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
print(f"Logs will be saved in: {log_file}")

# =====================
# Paths and Device Setup
# =====================
model_path = "distilroberta_waf_model"  # Path to your model
dataset_path = "../dataset/payload_test.csv"  # Path to the dataset

print(f"Model path: {model_path}")
print(f"Dataset path: {dataset_path}")

# Check paths
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory '{model_path}' not found.")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================
# Load Model and Tokenizer
# =====================
print("\nLoading DistilRoBERTa model and tokenizer...")
try:
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading model from '{model_path}': {e}")
model.to(device)
print("Model and tokenizer loaded successfully.")

# =====================
# Label Mapping
# =====================
label_mapping = {'LEGAL': 0, 'XSS': 1, 'SQL': 2, 'SHELL': 3}  # Ensure these match the model's training
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Attack type mapping (adjust to your data)
attack_type_mapping = {
    'norm': 'LEGAL',
    'sqli': 'SQL',
    'xss': 'XSS',
    'cmdi': 'SHELL'
}

# =====================
# Load Dataset
# =====================
def load_dataset(dataset_path):
    """
    Load dataset from the specified path, map attack_type labels, and exclude unwanted types.
    """
    data = pd.read_csv(dataset_path)

    if 'payload' not in data.columns or 'attack_type' not in data.columns:
        raise ValueError("The dataset must contain 'payload' and 'attack_type' columns.")

    print(f"Initial dataset size: {len(data)} rows.")
    data['attack_type'] = data['attack_type'].map(attack_type_mapping)  # Map attack types

    # Exclude rows with unmapped labels
    data = data.dropna(subset=['attack_type'])
    print(f"Filtered dataset size: {len(data)} rows (after excluding unmapped labels).")

    # Convert attack_type to indices
    data['attack_type'] = data['attack_type'].map(label_mapping)

    return data

# =====================
# Preprocess Payload
# =====================
def preprocess_payload(payload):
    """
    Preprocess the payload using the tokenizer and return encodings.
    """
    encodings = tokenizer(
        payload,
        truncation=True,
        padding=True,
        max_length=128,  # Adjust if payloads are longer
        return_tensors="pt"
    )
    return encodings["input_ids"].to(device), encodings["attention_mask"].to(device)

# =====================
# Predict Payload Class
# =====================
def predict_payload(payload):
    """
    Predict the class of a given payload.
    """
    input_ids, attention_mask = preprocess_payload(payload)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        confidence_scores = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_class_idx = logits.argmax(dim=1).item()
    predicted_class = reverse_label_mapping[predicted_class_idx]
    confidence_score = confidence_scores[predicted_class_idx]

    return predicted_class_idx, predicted_class, confidence_score

# =====================
# Evaluate Dataset
# =====================
def evaluate_dataset(dataset_path, output_false_file):
    """
    Evaluate the DistilRoBERTa model on the given dataset and save false predictions.
    """
    data = load_dataset(dataset_path)

    # Print unique labels in the dataset
    unique_labels = data['attack_type'].unique()
    print("\nUnique labels in the dataset:")
    for label_idx in unique_labels:
        print(f"  - {reverse_label_mapping[label_idx]}")

    predictions = []
    true_labels = data['attack_type'].tolist()
    false_payloads = []

    tp, tn, fp, fn = 0, 0, 0, 0

    # Process the dataset
    print("\nProcessing dataset...")
    for _, row in tqdm(data.iterrows(), total=len(data)):
        payload = row['payload']
        true_label_idx = row['attack_type']  # Already mapped to indices

        predicted_class_idx, predicted_class, confidence_score = predict_payload(payload)
        predictions.append(predicted_class_idx)

        # Track performance metrics
        if predicted_class_idx == true_label_idx:
            if predicted_class_idx == label_mapping['LEGAL']:
                tn += 1
            else:
                tp += 1
        else:
            if predicted_class_idx == label_mapping['LEGAL']:
                fn += 1
            else:
                fp += 1
            false_payloads.append({
                'payload': payload,
                'true_label': reverse_label_mapping[true_label_idx],
                'predicted_label': predicted_class,
                'confidence_score': confidence_score
            })

    # Calculate metrics
    total = tp + tn + fp + fn
    acc = (tp + tn) / total * 100 if total > 0 else 0
    precision = precision_score(true_labels, predictions, average='weighted') * 100
    recall = recall_score(true_labels, predictions, average='weighted') * 100
    fpr = fp / (fp + tn) * 100
    fnr = fn / (fn + tp) * 100
    f1 = f1_score(true_labels, predictions, average="weighted") * 100

    # Print metrics
    print(f"\nMetrics Summary:")
    print(f"  True Positives: {tp}\nTrue Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    print(f"  F1-Score: {f1:.2f}%")
    print(f"  False Positive Rate: {fpr: .2f}%")
    print(f"  False Negative Rate: {fnr: .2f}%")

    logging.info(f"Metrics: Accuracy={acc:.2f}%, Precision={precision:.2f}%, Recall (TPR)={recall:.2f}%, False Positive Rate: {fpr: .2f}%, F1={f1:.2f}% False Negative Rate: {fnr: .2f}%, F1={f1:.2f}%")

    # Save false payloads
    false_payloads_df = pd.DataFrame(false_payloads)
    false_payloads_df.to_csv(output_false_file, index=False)
    print(f"\nFalse payloads saved to: {output_false_file}")
    logging.info(f"False payloads saved to: {output_false_file}")

# =====================
# Main Entry Point
# =====================
if __name__ == "__main__":
    evaluate_dataset(dataset_path, output_false_file)
