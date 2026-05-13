import os
import json
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW
from tqdm import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
epochs = 5
learning_rate = 5e-5

model_name = "Salesforce/blip-image-captioning-base"

json_path = "annotations\\annotations_annotator_01.json"
images_dir = "images"

save_directory = "fine_tuned_models\\BLIP"
os.makedirs(save_directory, exist_ok=True)

print(f"Loading BLIP Processor from: {model_name}")
processor = BlipProcessor.from_pretrained(model_name)

def load_captions(path):    
    with open(path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
        
    samples_list = []
    
    for item in json_data:
        sample_id = item.get("sample_id")
        description = item.get("description")
        
        if isinstance(description, str):
            if description.strip() != "" and description != "NaN":
                samples_list.append({
                    "sample_id": sample_id,
                    "description": description
                })
                
    print(f"Successfully processed {len(samples_list)} valid text training targets.")
    return samples_list

class StructuralCaptionDataset(Dataset):
    def __init__(self, samples, img_dir):
        self.samples = samples
        self.img_dir = img_dir
        self.valid_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        description = sample["description"]
        
        image_path = None
        for extension in self.valid_extensions:
            test_path = os.path.join(self.img_dir, f"{sample_id}{extension}")
            if os.path.exists(test_path):
                image_path = test_path
                break
                
        if image_path is None:
            raise FileNotFoundError(f"Could not find an image file for sample: {sample_id}")
            
        img = Image.open(image_path).convert("RGB")
        return img, description

def collate(batch):
    images = []
    texts = []
    
    for item in batch:
        images.append(item[0])
        texts.append(item[1])
        
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def train_vlm_pipeline():
    train_samples = load_captions(json_path)
    
    dataset = StructuralCaptionDataset(train_samples, images_dir)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate
    )
    
    print(f"Downloading weights for: {model_name}")
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    print(f"\nStarting fine-tuning sequence for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} of {epochs}")
        
        for batch in progress_bar:
            gpu_batch = {}
            for key, tensor_value in batch.items():
                gpu_batch[key] = tensor_value.to(device)
                
            optimizer.zero_grad()
            
            outputs = model(**gpu_batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss = total_loss + loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        average_epoch_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1} Complete | Average Training Loss: {average_epoch_loss:.4f}")
        
    print(f"\nSaving model states and processor files to: {save_directory}")
    model.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)

if __name__ == "__main__":
    train_vlm_pipeline()