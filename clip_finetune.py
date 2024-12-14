import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from PIL import Image
import os
import json

# Configuration
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 5
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)

total_params = sum(p.numel() for p in model.parameters())


print(f"Total parameters: {total_params}")

torch.cuda.empty_cache()
from torch.utils.data import Dataset
from PIL import Image
class MapLMDataset(Dataset):
    def __init__(self, annot_data,processor):
      self.annot_data=annot_data
      self.data=[]
      self.processor = processor
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
                image = image.convert("RGB")  # Convert grayscale to RGB
            images.append(image)
        processed_images = self.processor(images=images, return_tensors="pt")
        return processed_images["pixel_values"]


    
    def generate_examples(self):
        
        for i, anno in enumerate(self.annot_data):
            # print(anno["images"])
            pixel_values = self.preprocess_images( anno["images"])
            for c in anno["conversations"]:
                if c["question_type"] in ["SCN","QLT","INT","LAN"]:
                    options=c["options"]
                    question=c["question"]
                    label=c["answer"]

                    # # yield i, data_item
                    # self.data.append(data_item)
                    options_text = "\n".join([f"{opt}" for i, opt in enumerate(options)])
                    prompt = f"Context: The images depict certain scenes.\nQuestion: {question}\nOptions:\n{options_text}\nAnswer:"
                    

                    self.data.append({
                        "images": pixel_values,
                        "prompt": prompt,
                        "label": label,  # Convert 'A' -> 0, 'B' -> 1, etc.
                    }) 

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
def collate_fn(batch):
    images = torch.cat([sample["images"] for sample in batch])  # Concatenate images into a single tensor
    prompts = [sample["prompt"] for sample in batch]
    labels = torch.tensor([sample["label"] for sample in batch])
    return {"images": images, "prompt": prompts, "label": labels}

# Training Loop
def train_one_epoch(model, dataloader, optimizer, processor, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        # Preprocess images and prompts
        images = batch["images"].to(device) 
        # images = [image for sample in batch["images"] for image in sample]
        prompts = batch["prompt"]
        labels = batch["label"].to(device)

        # Process text only
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        inputs["pixel_values"] = images  # Add pixel_values manually

        # Forward pass
        outputs = model(**inputs)
        logits_per_text = outputs.logits_per_text  # Shape: (batch_size, num_classes)

        # Compute loss
        loss = torch.nn.CrossEntropyLoss()(logits_per_text, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Training Loss: {avg_loss:.4f}")
    return model,avg_loss


# Validation Loop
def validate(model, dataloader, processor, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss=0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Preprocess images and prompts
            images = batch["images"].to(device) 
            # images = [image for sample in batch["images"] for image in sample]
            prompts = batch["prompt"]
            labels = batch["label"].to(device)

            # Process text only
            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            inputs["pixel_values"] = images  # Add pixel_values manually


            # Forward pass
            outputs = model(**inputs)
            logits_per_text = outputs.logits_per_text  # Shape: (batch_size, num_classes)

            # Get predictions
            preds = torch.argmax(logits_per_text, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = torch.nn.CrossEntropyLoss()(logits_per_text, labels)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader.dataset)

    # Compute metrics
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"Loss: {avg_loss:.4f},Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    #return precision, recall, f1
    return model,avg_loss,accuracy
def save_model_checkpoint(model, optimizer, epoch, loss, checkpoint_path='model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}")
from transformers import AdamW

# Prepare data and model
# data = load_data("path_to_data.json")  # Replace with actual data loading logic
# processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6")
# train_loader, val_loader = get_dataloaders(data, processor, batch_size=8)

# Model and optimizer
# model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6").to(device)
data_path = "data/test/test.json"  # Replace with the actual path to the dataset file
with open(data_path, 'r') as file:
        annot_data = json.load(file)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
test_dataset = MapLMDataset(annot_data, processor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
# train_dataset = MapLMDataset(annot_data, processor)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
# print(len(train_loader.dataset))
# with open(data_path.replace("train", "val"), 'r') as file:
#         annot_data = json.load(file)
# val_dataset = MapLMDataset(annot_data, processor)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
# print(len(train_loader.dataset))
optimizer = AdamW(model.parameters(), lr=1e-4)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Step down LR every 5 epochs
# Training loop
train=False
if train:
    epochs = 10
    best_accuracy=0
    patience=3
    patience_counter=0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model,total_loss=train_one_epoch(model, train_loader, optimizer, processor, device)
        model,val_loss,val_accuracy=validate(model, val_loader, processor, device)
        save_model_checkpoint(model, optimizer, epoch, total_loss, checkpoint_path="exp/clip1/"+str(epoch)+'_model_checkpoint.pth')
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            save_model_checkpoint(
                model, optimizer, epoch, total_loss, checkpoint_path=f"exp/clip1/best_model_checkpoint.pth"
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
else:
    # test goes here
    #
    # Load the trained model checkpoint
    checkpoint_path = "/raid/ai22resch01001/exp/0_model_checkpoint.pth"  # Replace with your best model checkpoint path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate on test dataset
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["images"].to(device)
            prompts = batch["prompt"]
            labels = batch["label"].to(device)

            # Process prompts with the processor
            inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            inputs["pixel_values"] = images

            # Forward pass
            outputs = model(**inputs)
            logits_per_text = outputs.logits_per_text  # Shape: (batch_size, num_classes)

            # Predict labels
            preds = torch.argmax(logits_per_text, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute evaluation metrics
        accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        precision = precision_score(all_labels, all_preds, average="macro")
        recall = recall_score(all_labels, all_preds, average="macro")
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
                
