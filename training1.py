import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tqdm import tqdm

#  Define transformations (Resizing, Normalization, Data Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Keep resizing
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Keep normalization
])

#  Load dataset using folder names as labels
dataset_path = "crop_resized"  # Update with your dataset path
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

#  Count class distribution
train_class_counts = Counter([dataset.targets[i] for i in train_dataset.indices])
val_class_counts = Counter([dataset.targets[i] for i in val_dataset.indices])

# Convert class indices to class names
train_class_counts = {dataset.classes[k]: v for k, v in train_class_counts.items()}
val_class_counts = {dataset.classes[k]: v for k, v in val_class_counts.items()}

print("\n Train Class Distribution:", train_class_counts)
print("� Validation Class Distribution:", val_class_counts)

# Check if class imbalance exists (Threshold: 5x difference)
max_count = max(train_class_counts.values())
min_count = min(train_class_counts.values())
imbalance_threshold = 5  # If max count is 5x or more than min count, it's imbalanced

use_weighted_sampler = (max_count / min_count) >= imbalance_threshold

if use_weighted_sampler:
    print("\n⚠ Class Imbalance Detected! Applying Weighted Random Sampling...")
    
    #  FIX: Properly map dataset class indices
    class_weights = {i: 1.0 / train_class_counts[dataset.classes[i]] for i in range(len(dataset.classes))}
    
    sample_weights = [class_weights[dataset.targets[idx]] for idx in train_dataset.indices]  #  Corrected Mapping!
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)  #  Uses sampler!
else:
    print("\n No Major Class Imbalance. Using Regular Training.")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation Loader (Always Regular)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#  Get class names (from folder names)
class_names = dataset.classes
print(f"\n🎯 Detected {len(class_names)} classes: {class_names}")

#  Load Pretrained ResNet-18
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

#  Modify the last layer to match the number of aircraft classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Adjust output layer

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#  Define Loss & Optimizer (Lower Learning Rate + Weight Decay)
criterion = nn.CrossEntropyLoss()  # Multi-class classification loss
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)  # Adam optimizer

#  Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_acc = 0.0  # Track best accuracy
    
    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="🔄 Training", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f" Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="📊 Validating", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f" Validation Acc: {val_acc:.4f}")

        # Save best model with num_classes
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'num_classes': len(class_names)}, "best_aircraft_model.pth")
            print(" Model Saved!")

    print(f"\n Training Completed! Best Validation Accuracy: {best_acc:.4f}")

#  Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

#  Load the trained model
checkpoint = torch.load("best_aircraft_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#  Evaluate CNN performance on validation data
y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="📊 Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

#  Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.show()

#  Print Classification Report
print("\n📌 Classification Report:\n", classification_report(y_true, y_pred, target_names=class_names))
