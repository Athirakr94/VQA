import os
import json
import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
def read_json(mode):
    file_path=os.getcwd()+"/data/"+mode+"/"+mode+".json"
    annot_data=None
    with open(file_path, 'r') as file:
        annot_data = json.load(file)
    return annot_data
def get_data(mode):
    annot_data=read_json(mode)
    return annot_data
class MapLMDataset(Dataset):
    def __init__(self, mode,annot_data):
      self.mode=mode
      self.annot_data=annot_data
      self.data=[]
      self.generate_examples()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      return self.data[idx]
    def generate_examples(self):
        for i, anno in enumerate(self.annot_data):
            data_item = {}
            data_item["frame_id"] = anno["frame_id"]
            data_item["images"] = anno["images"]
            data_item["question"] = []
            data_item["options"] = []
            data_item["answer"] = []
            data_item["question_type"] = []
            for c in anno["conversations"]:
                if c["question_type"] in ["SCN","QLT","INT","LAN"]:
                
                    data_item["question"].append(c["question"])
                    data_item["options"].append(c["options"])
                    data_item["answer"].append(c["answer"])
                    data_item["question_type"].append(c["question_type"])
            # yield i, data_item
            self.data.append(data_item)
MAX_SEQ_LENGTH = 128
class VQADataset(Dataset):
    def __init__(self, dataset_path, transform, tokenizer):
        with open(dataset_path, "r") as f:
            self.data = json.load(f)

        self.transform = transform
        self.tokenizer = tokenizer
        self.max_options=0
        # Flatten conversations
        self.flat_data = []
        for item in self.data:
            
            for question in item["conversations"]:
                if question["question_type"] in ["SCN","QLT","INT","LAN"]:
                    self.flat_data.append({
                        "images": item["images"],
                        "question": question["question"],
                        "options": question.get("options", []),  # Default to an empty list for Q&A
                        "answer": question["answer"],
                        "type": question["question_type"]
                    })

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        # Retrieve flattened question
        item = self.flat_data[idx]
        #print(item)
        # Load and transform images
        left_image = self.transform(Image.open(item["images"][0]).convert("RGB"))
        right_image = self.transform(Image.open(item["images"][1]).convert("RGB"))
        center_image = self.transform(Image.open(item["images"][2]).convert("RGB"))
        bev_image = self.transform(Image.open(item["images"][3]).convert("RGB"))

        # Tokenize question and options (if any)
        question = item["question"]
        options = item["options"]
        question_type = item["type"]
        if len(options)>self.max_options:
            self.max_options=len(options)
        if question_type in ["SCN", "QLT", "INT", "LAN"] and len(options) > 0:
            encoded = self.tokenizer(
                [question] * len(options),  # Repeat question for each option
                options,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        label = int(item["answer"])
        '''else:  # For open-ended Q&A
            encoded = self.tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            # print(item["answer"])
            label = self.tokenizer(str(item["answer"]), padding="max_length", truncation=True, max_length=128, return_tensors="pt")["input_ids"].squeeze(0)
'''
        #label = item["answer"]
        #print(self.max_options)
        return {
            "images": (left_image, right_image, center_image, bev_image),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": label,
            "type": question_type
        }



def custom_collate_fn(batch):
    # Filter out None items
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        raise ValueError("All items in the batch are invalid.")

    images = {"left": [], "right": [], "center": [], "bev": []}
    input_ids = []
    attention_mask = []
    mcq_labels = []
    question_types = []

    for item in batch:
        # Append images
        left, right, center, bev = item["images"]
        images["left"].append(left)
        images["right"].append(right)
        images["center"].append(center)
        images["bev"].append(bev)

        # Append input IDs and attention masks
        input_ids.append(item["input_ids"])  # List of tensors
        attention_mask.append(item["attention_mask"])  # List of tensors

        # Append labels and question types
        mcq_labels.append(int(item["label"]))
        question_types.append(item["type"])

    # Stack image tensors
    images["left"] = torch.stack(images["left"])
    images["right"] = torch.stack(images["right"])
    images["center"] = torch.stack(images["center"])
    images["bev"] = torch.stack(images["bev"])

    # Pad input_ids and attention_mask dynamically across all dimensions
    max_num_options = max(seq.size(0) for seq in input_ids)
    max_seq_len = max(seq.size(1) for seq in input_ids)
    input_ids_padded = torch.zeros((len(batch), max_num_options, max_seq_len), dtype=torch.long)
    attention_mask_padded = torch.zeros((len(batch), max_num_options, max_seq_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        input_ids_padded[i, :ids.size(0), :ids.size(1)] = ids
        attention_mask_padded[i, :mask.size(0), :mask.size(1)] = mask

    # Convert labels to tensor
    mcq_labels = torch.tensor(mcq_labels, dtype=torch.long)

    return {
        "images": images,
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "mcq_labels": mcq_labels,
        "question_types": question_types,  # Keep as list of strings for dynamic handling
    }


def prepare_dataset(mode,batch_size):
    annot_data=get_data(mode)
    # print(annot_data)
    dataset_path= os.getcwd()+"/data/"+mode+"/"+mode+".json"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset=VQADataset(dataset_path, transform,tokenizer)
    #dataset=MapLMDataset(mode,annot_data)
    dataloader=DataLoader(dataset, shuffle=True, batch_size=batch_size,collate_fn=custom_collate_fn)
    return dataloader
def get_dataset(mode,batch_size):
    annot_data=get_data(mode)
    # print(annot_data)
    dataset_path= os.getcwd()+"/data/"+mode+"/"+mode+".json"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #dataset=VQADataset(dataset_path, transform,tokenizer)
    dataset=MapLMDataset(mode,annot_data)
    dataloader=DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader
if __name__ == "__main__": 
    
    batch_size=16
    train_dataloader=prepare_dataset("train",batch_size)
    val_dataloader=prepare_dataset("val",batch_size)
    test_dataloader=prepare_dataset("test",batch_size)
    print(len(train_dataloader.dataset))
    print(len(val_dataloader.dataset))
    print(len(test_dataloader.dataset))
    for data in train_dataloader:
        print(data)



