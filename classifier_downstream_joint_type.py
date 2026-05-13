import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
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
batch_size = 32
epochs = 50
learning_rate = 0.0001

json_path = "annotations\\annotations_annotator_01.json"
images_dir = "images"


def load_labels_from_json(path):

    print(f"Loading annotations from: {path}")
    
    with open(path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
        
    unique_joint_types = []
    for item in json_data:
        joint_type = item.get("joint_type")
        if joint_type and joint_type not in unique_joint_types:
            unique_joint_types.append(joint_type)
            
    unique_joint_types.sort()
    
    label_mapping = {}
    for index, joint_type in enumerate(unique_joint_types):
        label_mapping[joint_type] = index
        
    print("\nJoint Type Mapping:")
    for joint_type, label_int in label_mapping.items():
        print(f"  Class {label_int}: {joint_type}")
        
    samples_list = []
    for item in json_data:
        sample_id = item.get("sample_id")
        joint_type = item.get("joint_type")
        
        if joint_type in label_mapping:
            label_int = label_mapping[joint_type]
            samples_list.append({
                "sample_id": sample_id,
                "label": label_int
            })
            
    return samples_list, label_mapping


class StructuralDamageDataset(Dataset):
    def __init__(self, samples, img_dir, transform=None):
        self.samples = samples
        self.img_dir = img_dir
        self.transform = transform
        self.valid_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["sample_id"]
        label = sample["label"]
        
        image_path = None
        for extension in self.valid_extensions:
            test_path = os.path.join(self.img_dir, f"{sample_id}{extension}")
            if os.path.exists(test_path):
                image_path = test_path
                break
                
        if image_path is None:
            raise FileNotFoundError(f"Could not find an image file for sample: {sample_id}")
            
        img = Image.open(image_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)



def evaluate_model(model, data_loader):
    model.eval()
    
    true_labels = []
    predicted_labels = []
    predicted_probabilities = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, dim=1)
            
            # Transfer data cleanly back to CPU arrays
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            predicted_probabilities.extend(probabilities.cpu().numpy())
            
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probabilities = np.array(predicted_probabilities)
    
    accuracy = accuracy_score(true_labels, predicted_labels) * 100
    f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0) * 100
    
    auc = roc_auc_score(
        true_labels, 
        predicted_probabilities, 
        multi_class="ovr", 
        average="macro"
    )

    return accuracy, f1, auc


def train_pipeline():
    raw_samples, label_mapping = load_labels_from_json(json_path)
    num_classes = len(label_mapping)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_labels = []
    for sample in raw_samples:
        all_labels.append(sample["label"])
        
    train_samples, test_samples = train_test_split(
        raw_samples,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    
    print(f"\nDataset Successfully Split:")
    print(f"Training samples: {len(train_samples)}")
    print(f"Testing samples:  {len(test_samples)}\n")
    
    # Initialize separate Dataset classes
    train_dataset = StructuralDamageDataset(train_samples, images_dir, transform=train_transform)
    test_dataset = StructuralDamageDataset(test_samples, images_dir, transform=test_transform)
    
    train_labels = []
    for sample in train_samples:
        train_labels.append(sample["label"])
        
    unique_classes = np.array(list(label_mapping.values()))
    weights = compute_class_weight("balanced", classes=unique_classes, y=train_labels)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    
    # Setup standard DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model once using the dynamic number of discovered classes
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    accuracy, f1_score_val, auc_score = evaluate_model(model, test_loader)

    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"F1 Score:  {f1_score_val:.2f}%")
    print(f"AUC Score: {auc_score:.4f}")


if __name__ == "__main__":
    train_pipeline()