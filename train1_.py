import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import json
from torch.optim import Adam
from dataloader import prepare_dataset
import tqdm
from sklearn.metrics import accuracy_score
import time
# === Configuration ===
NUM_CLASSES = 6  # Number of answer choices per question
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 50
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# === Model ===
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision import models
import torch.nn.functional as F
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
class VQAModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', num_mcq_options=4, hidden_size=768, dropout_rate=0.1):
        super(VQAModel, self).__init__()
        
        # Load pretrained text model (BERT)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        
        # Freeze text encoder layers (optional)
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False
        
        # Image Encoder: ResNet (you can replace with ViT or any other CNN)
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Identity()  # Remove the final classification layer

        # Fusion layer: Combine text and image features
        self.fc_text = nn.Linear(self.text_encoder.config.hidden_size, hidden_size)
        self.fc_image = nn.Linear(2048, hidden_size)  # ResNet50 output features (2048)

        # MCQ Head (for multiple choice questions)
        self.mcq_head = nn.Linear(2 * hidden_size, num_mcq_options)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, left, right, center, bev, input_ids, attention_mask):
        # Step 1: Process the images through the image encoder
        left_features = self.image_encoder(left)
        right_features = self.image_encoder(right)
        center_features = self.image_encoder(center)
        bev_features = self.image_encoder(bev)

        # Compute the mean image features across the four images (Shape: [batch_size, 2048])
        image_features = (left_features + right_features + center_features + bev_features) / 4
        image_features = self.fc_image(image_features)  # Shape: [batch_size, hidden_size]

        # Compute the mean image features across the four images
        #image_features = (left_features + right_features + center_features + bev_features) / 4  # Shape: [batch_size, 2048]
        #image_features = self.fc_image(image_features)  # Shape: [batch_size, hidden_size]

        # Step 2: Flatten inputs for multiple-choice questions
        batch_size, num_options, seq_length = input_ids.size()
        input_ids = input_ids.view(-1, seq_length)  # Shape: [batch_size * num_options, seq_length]
        attention_mask = attention_mask.view(-1, seq_length)  # Shape: [batch_size * num_options, seq_length]

        # Encode text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # Shape: [batch_size * num_options, hidden_size]

        # Step 3: Reshape text features to [batch_size, num_options, hidden_size]
        text_features = text_features.view(batch_size, num_options, -1)  # Shape: [batch_size, num_options, hidden_size]
        text_features = self.fc_text(text_features)  # Shape: [batch_size, num_options, hidden_size]

        # Step 4: Fuse features: Concatenate image and text features
        # Concatenate along the feature dimension (last dimension)
        fused_features = torch.cat((image_features.unsqueeze(1).expand(-1, num_options, -1), text_features), dim=-1)
        fused_features = self.dropout(fused_features)  # Apply dropout

        # Step 5: MCQ logits: Classification head for multiple-choice
        mcq_logits = self.mcq_head(fused_features)  # Shape: [batch_size, num_options]
        mcq_logits = mcq_logits.squeeze(1)  # Remove the unnecessary dimension, Shape: [batch_size, num_options]
        mcq_logits = mcq_logits.mean(dim=1) 
        return mcq_logits

# Decoding function for Q&A answers (you may use more complex decoding strategies)
def decode_predictions(predictions, tokenizer):
    decoded = []
    for prediction in predictions:
        decoded_text = tokenizer.decode(prediction, skip_special_tokens=True)
        decoded.append(decoded_text)
    return decoded

# Simple evaluation for Q&A (e.g., matching predictions with ground truth)
def evaluate_qa_answers(true_answers, predicted_answers):
    correct = 0
    total = len(true_answers)
    for true_answer, predicted_answer in zip(true_answers, predicted_answers):
        if true_answer.strip().lower() == predicted_answer.strip().lower():  # Case insensitive comparison
            correct += 1
    return correct / total if total > 0 else 0
def save_model_checkpoint(model, optimizer, epoch, loss, checkpoint_path='model_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}")
def load_model_checkpoint(model, optimizer, checkpoint_path='model_checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from checkpoint at epoch {epoch} with loss {loss:.4f}")
    return model, optimizer, epoch, loss
# === Main Script ===
if __name__ == "__main__":
    # Transform and Tokenizer
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Dataset and DataLoader
    batch_size=16
    mode="train"
    train_loader=prepare_dataset(mode,batch_size)
    mode="val"
    val_loader=prepare_dataset(mode,batch_size)
    # Model, Optimizer, and Loss Function
    model = VQAModel(num_mcq_options=6).to(device)  # Set the number of MCQ options
    #optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()  # For MCQs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


    
    for epoch in range(EPOCHS):
        t1=time.time()
        total_loss = 0
        total_mcq_accuracy = 0
        #total_qa_accuracy = 0
        num_batches = len(train_loader)
        # Training loop
        model.train()
        for batch in train_loader:
            # Move data to device
            left = batch['images']["left"].to(device)
            right= batch['images']["right"].to(device)
            center = batch['images']["center"].to(device)
            bev=batch['images']["bev"].to(device)
            #images = batch["images"].to(device)  # Move all image tensors
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mcq_labels = batch["mcq_labels"].to(device)

            # Combine images into a single tensor
            # Assuming image features need to be averaged or concatenated
            #image_batch = torch.cat([images["left"], images["right"], images["center"], images["bev"]], dim=1)

            # Forward pass
            optimizer.zero_grad()
            mcq_logits = model(left,right,center,bev, input_ids, attention_mask)
            assert mcq_labels.max() < NUM_CLASSES and mcq_labels.min() >= 0, f"Invalid label: {mcq_labels.max()} or {mcq_labels.min()}"
            # Compute loss
            #print(mcq_logits.shape,mcq_labels.shape)
            loss = criterion(mcq_logits, mcq_labels)
            total_loss += loss.item()
            mcq_accuracy = (torch.argmax(mcq_logits, dim=-1) == mcq_labels).float().mean()
            total_mcq_accuracy += mcq_accuracy.item()
            
            # Handle Q&A
            '''elif question_type == "qa":
                qa_logits = model(images[i], input_ids[i], attention_mask[i], question_type="qa")
                qa_loss = F.mse_loss(qa_logits, qa_labels[i])  # Use appropriate loss for Q&A
                qa_loss.backward()
                total_loss += qa_loss.item()
                # Example accuracy for Q&A (use BLEU or another metric for better evaluation)
                qa_accuracy = (qa_logits.argmax(dim=-1) == qa_labels[i].argmax(dim=-1)).float().mean()
                total_qa_accuracy += qa_accuracy.item()'''

            optimizer.step()

        # Print loss and accuracy after each epoch
        print(f"Train Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader.dataset):.4f}, "
              f"MCQ Accuracy: {total_mcq_accuracy/len(train_loader.dataset):.4f}")
        # Saving the model after every epoch
        save_model_checkpoint(model, optimizer, epoch, total_loss/len(train_loader.dataset), checkpoint_path="exp/exp1_/"+str(epoch)+'_model_checkpoint.pth')
        
        # Loop through the test data
        model.eval()
        total_mcq_accuracy = 0
        #total_qa_accuracy = 0
        mcq_true_labels = []
        mcq_pred_labels = []
        #qa_true_answers = []
        total_loss=0
        #qa_pred_answers = []
        
        with torch.no_grad():
            for batch in val_loader:
                left = batch['images']["left"].to(device)
                right= batch['images']["right"].to(device)
                center = batch['images']["center"].to(device)
                bev=batch['images']["bev"].to(device)
                #images = batch["images"].to(device)  # Move all image tensors
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                mcq_labels = batch["mcq_labels"].to(device)

                # Combine images into a single tensor
                # Assuming image features need to be averaged or concatenated
                #image_batch = torch.cat([images["left"], images["right"], images["center"], images["bev"]], dim=1)

                # Forward pass
                optimizer.zero_grad()
                mcq_logits = model(left,right,center,bev, input_ids, attention_mask)
                assert mcq_labels.max() < NUM_CLASSES and mcq_labels.min() >= 0, f"Invalid label: {mcq_labels.max()} or {mcq_labels.min()}"
                # Compute loss
                #print(mcq_logits.shape,mcq_labels.shape)
                loss = criterion(mcq_logits, mcq_labels)
                total_loss += loss.item()
                mcq_accuracy = (torch.argmax(mcq_logits, dim=-1) == mcq_labels).float().mean()
                total_mcq_accuracy += mcq_accuracy.item()
                
                
            
        print(f"Val Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader.dataset):.4f}, "
              f"MCQ Accuracy: {total_mcq_accuracy/len(train_loader.dataset):.4f}")        ##print(f"Q&A Accuracy: {qa_accuracy * 100:.2f}%")
        print("Time taken for epoch ",epoch," is ",time.time()-t1)
        # if validation_accuracy > best_validation_accuracy:
        #     best_validation_accuracy = validation_accuracy
        #     save_model_checkpoint(model, optimizer, epoch, total_loss/num_batches, checkpoint_path)



    

    
    infer=False
    # model, optimizer, epoch, loss = load_model_checkpoint(model, optimizer, checkpoint_path='model_checkpoint.pth')
    # if infer:
    # # Example Inference
    #     model.eval()

    #     # For a sample input
    #     sample_image = sample_image_tensor
    #     sample_input_ids = sample_question_ids
    #     sample_attention_mask = sample_attention_mask

    #     # MCQ Inference
    #     mcq_logits = model(sample_image, sample_input_ids, sample_attention_mask, question_type="mcq")
    #     mcq_predictions = torch.argmax(mcq_logits, dim=-1)  # Predicted option

    #     # Q&A Inference
    #     qa_logits = model(sample_image, sample_input_ids, sample_attention_mask, question_type="qa")
    #     qa_predictions = qa_logits  # Predicted answer

