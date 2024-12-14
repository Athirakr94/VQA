import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from PIL import Image
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import random

# Dataset
class MapLMDataset(Dataset):
    def __init__(self, annot_data, processor, tokenizer, max_options=6):
        self.annot_data = annot_data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_options = max_options
        self.data = []
        self.generate_examples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess_images(self, image_paths):
        images = []
        for path in image_paths:
            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)
        processed_images = self.processor(images=images, return_tensors="pt")
        return processed_images["pixel_values"]

    def pad_and_mask_options(self, options):
        padded_options = options + [''] * (self.max_options - len(options))
        mask = [1] * len(options) + [0] * (self.max_options - len(options))
        return padded_options, mask

    def generate_examples(self):
        for anno in self.annot_data:
            pixel_values = self.preprocess_images(anno["images"])
            for c in anno["conversations"]:
                if c["question_type"] in ["SCN", "QLT", "INT", "LAN"]:
                    options = c["options"]
                    question = c["question"]
                    label = c["answer"]
                    padded_options, mask = self.pad_and_mask_options(options)
                    options_text = "\n".join([f"{opt}" for opt in padded_options])
                    prompt = f"Context: The images depict certain scenes.\nQuestion: {question}\nOptions:\n{options_text}\nAnswer:"
                    self.data.append({
                        "images": pixel_values,
                        "prompt": prompt,
                        "label": label,
                        "mask": mask
                    })

def collate_fn(batch):
    pixel_values = torch.stack([item["images"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    masks = torch.tensor([item["mask"] for item in batch])  # (batch_size, num_options)
    return {"images": pixel_values, "prompt": prompts, "label": labels, "mask": masks}
#  Load data
with open("data/train/train.json", 'r') as f:
    train_data = json.load(f)
with open("data/val/val.json", 'r') as f:
    val_data = json.load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")



# Model
from transformers import ViTForImageClassification, BertForSequenceClassification

class VQA_ViT_BERT(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224-in21k", bert_model_name="bert-base-uncased", num_options=6):
        super(VQA_ViT_BERT, self).__init__()

        # Vision Transformer (ViT) for image processing
        self.vit = ViTForImageClassification.from_pretrained(vit_model_name)

        # BERT model for question answering (classification)
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_options)

        # Linear layer to combine ViT and BERT outputs
        self.fc = nn.Linear(self.vit.config.hidden_size + self.bert.config.hidden_size, num_options)

    def forward(self, images, input_ids, attention_mask, mask=None):
        # Forward pass through ViT for image embeddings
        vit_output = self.vit(pixel_values=images).logits
        
        # Forward pass through BERT for question embeddings, use attention_mask if provided
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

        # Apply option mask if provided
        if mask is not None:
            # Ensure mask matches the size of bert_output
            mask = mask[:, :bert_output.size(1)]  # Adjust dimensions if necessary
            bert_output = bert_output * mask.float()  # Apply the mask

        # Concatenate both ViT and BERT outputs
        combined_features = torch.cat((vit_output, bert_output), dim=1)

        
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = VQA_ViT_BERT(
    vit_model_name="google/vit-base-patch16-224-in21k",
    bert_model_name="bert-base-uncased",
    num_options=6
).to(device)

# Hyperparameters
learning_rate = 1e-5
batch_size = 16
epochs = 20
weight_decay = 0.01

# Optimizer and Loss
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# Optional Learning Rate Scheduler
# from torch.optim.lr_scheduler import StepLR
# scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Print model summary
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
train_dataset = MapLMDataset(train_data, processor, tokenizer, max_options=6)
val_dataset = MapLMDataset(val_data, processor, tokenizer, max_options=6)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
def save_model_checkpoint(model, optimizer, epoch, loss, checkpoint_path='model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}")
print(f"Total Trainable Parameters: {total_params}")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch in train_loader:
        images = batch["images"].to(device)
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device)
        
        encoding = tokenizer(batch["prompt"], padding=True, truncation=True, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Forward pass
        logits = model(images, input_ids, attention_mask, mask)
        
        # Loss computation
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    train_loss = total_loss / len(train_loader.dataset)
    train_accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}")
    save_model_checkpoint(model, optimizer, epoch, train_loss, "exp/end/"+str(epoch)+".pth")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            labels = batch["label"].to(device)
            mask = batch["mask"].to(device)
            
            encoding = tokenizer(batch["prompt"], padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            
            logits = model(images, input_ids, attention_mask, mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
