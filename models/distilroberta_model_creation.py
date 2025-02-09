import pandas as pd
import torch
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import urllib.parse
import numpy as np

# =====================
# Step 1: Load and Inspect Dataset
# =====================
data_path = '../dataset/g_payloads.csv'  # Replace with your dataset path
df = pd.read_csv(data_path)

# Inspect dataset
print(f"Initial dataset shape: {df.shape}")
print(df.head())

# =====================
# Step 2: Clean and Encode Data
# =====================
df = df.dropna(subset=['payload', 'type'])
df['payload'] = df['payload'].astype(str).str.strip()
df = df[df['payload'] != '']  # Remove empty payloads
df['payload'] = df['payload'].apply(urllib.parse.unquote)

label_mapping = {label: idx for idx, label in enumerate(df['type'].unique())}
df['type'] = df['type'].map(label_mapping)
print(f"Dataset shape after cleaning: {df.shape}")
print(f"Label mapping: {label_mapping}")

if df.empty:
    raise ValueError("The dataset is empty after cleaning. Please check the input data.")

# =====================
# Step 3: Split Dataset into Training and Testing
# =====================
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['payload'], df['type'], test_size=0.2, random_state=42, stratify=df['type']
)

# =====================
# Step 4: Tokenize Data
# =====================
tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")  # Switched to distilroberta-base

def preprocess_data(payloads, labels, tokenizer, max_length=128):
    payloads = payloads.tolist()
    encodings = tokenizer(payloads, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels.tolist(), dtype=torch.long)
    }

train_data = preprocess_data(train_texts, train_labels, tokenizer)
test_data = preprocess_data(test_texts, test_labels, tokenizer)

print(f"Preprocessed dataset keys: {train_data.keys()}")

# ==========================
# Step 5: Create Dataset Class
# ==========================
class WAFDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

train_dataset = WAFDataset(train_data)
test_dataset = WAFDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# =====================
# Step 6: Define the Model
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
num_labels = len(label_mapping)
print(f"Number of labels: {num_labels}")

model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=num_labels)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

# Clear GPU memory cache before training
torch.cuda.empty_cache()

# =====================
# Step 7: Train the Model with Mixed Precision
# =====================
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

model.train()
for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loop.set_description(f'Epoch {epoch + 1}/{epochs}')
        loop.set_postfix(loss=(total_loss / (len(all_preds) + 1)))

    print(f"\nEpoch {epoch + 1} Metrics:")
    print(f"  Loss: {total_loss / len(train_loader):.4f}")

# =====================
# Step 8: Evaluate the Model
# =====================
def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []

    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))

# Evaluate the model
evaluate_model(model, test_loader)

# =====================
# Step 9: Save the Model
# =====================
print("Model training complete and saved.")
model.save_pretrained("./distilroberta_waf_model")
tokenizer.save_pretrained("./distilroberta_waf_model")
print("Model training complete and saved.")


