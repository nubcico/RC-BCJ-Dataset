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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
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
num_classes = 3

json_path = "annotations\\annotations_annotator_01.json"
images_dir = "images"

def load_labels_from_json(path):
    """
    Reads the JSON annotations file and maps the text damage locations
    to integer labels (0, 1, 2) for our 3-class classification problem.
    """
    print(f"Loading annotations from: {path}")
    
    with open(path, "r", encoding="utf-8") as file:
        json_data = json.load(file)
        
    label_mapping = {
        "Beam flexure near joint (B failure)": 0,
        "Beam flexure & Joint shear (BJ failure)": 1,
        "Joint shear (J failure)": 2
    }
    
    samples_list = []
    
    for item in json_data:
        sample_id = item.get("sample_id")
        damage_location = item.get("damage_location")
        
        # Check if the sample has a valid target location we track
        if damage_location in label_mapping:
            label_int = label_mapping[damage_location]
            samples_list.append({
                "sample_id": sample_id,
                "label": label_int
            })
            
    return samples_list


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
        
        # Locate the image file path by trying common image extensions
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
    raw_samples = load_labels_from_json(json_path)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = StructuralDamageDataset(raw_samples, images_dir, transform=train_transform)
    validation_dataset = StructuralDamageDataset(raw_samples, images_dir, transform=validation_transform)
    
    all_labels = []
    for sample in raw_samples:
        all_labels.append(sample["label"])
        
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_f1_scores = []
    fold_auc_scores = []
    
    print(f"\nStarting 10-Fold Cross-Validation Setup...")
    print(f"Targeting a single classifier architecture across all folds.\n")
    
    for fold, (train_indices, val_indices) in enumerate(k_fold.split(np.zeros(len(all_labels)), all_labels)):
        print(f"--- Running Fold {fold + 1} of 10 ---")
        
        train_subset = Subset(train_dataset, train_indices)
        validation_subset = Subset(validation_dataset, val_indices)
        
        fold_train_labels = []
        for index in train_indices:
            fold_train_labels.append(all_labels[index])
            
        unique_classes = np.array([0, 1, 2])
        weights = compute_class_weight("balanced", classes=unique_classes, y=fold_train_labels)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
        

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        for epoch in tqdm(range(epochs), desc=f"Fold {fold + 1} Training Steps"):
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        acc, f1, auc = evaluate_model(model, validation_loader)
        
        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        fold_auc_scores.append(auc)
        
    average_accuracy = np.mean(fold_accuracies)
    average_f1 = np.mean(fold_f1_scores)
    average_auc = np.mean(fold_auc_scores)
    
    print(f"Average Accuracy:   {average_accuracy:.2f}%")
    print(f"Average F1 Score:   {average_f1:.2f}%")
    print(f"Average AUC Score:  {average_auc:.4f}")


if __name__ == "__main__":
    train_pipeline()