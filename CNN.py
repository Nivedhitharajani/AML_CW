# import torch
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix
# from torchvision import transforms
# from PIL import Image, ImageEnhance
# import yaml


import torch
import torchvision
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets import ImageFolder

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Define transformations (Must match training settings)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet Normalization
])

# ✅ Load validation dataset
dataset_path = "crop_resized"
dataset = ImageFolder(root=dataset_path, transform=transform)
class_names = dataset.classes  # List of class names

# ✅ Split dataset (80% Train, 20% Validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# ✅ Create Validation DataLoader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ Load Trained Model (ResNet-18)
model = torchvision.models.resnet18(pretrained=False)  # Do NOT use pretrained=True here
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Adjust output layer for 81 classes
model = model.to(device)

# ✅ Load Best Model Weights
model.load_state_dict(torch.load("best_aircraft_model.pth"))
model.eval()  # Set to evaluation mode
print("✅ Model Loaded Successfully!")

# ✅ Evaluate Model on Validation Data
y_true = []
y_pred = []
pred_counter = Counter()

with torch.no_grad():  # Disable gradient calculation for inference
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        y_true.extend(labels.cpu().numpy())  # Store ground truth labels
        y_pred.extend(predicted.cpu().numpy())  # Store predicted labels

        pred_counter.update(predicted.cpu().numpy())  # Count class predictions

# ✅ Print Unique Predicted Classes
unique_classes_predicted = set(y_pred)
print(f"\n✅ Unique Classes Predicted: {len(unique_classes_predicted)} out of {len(class_names)}")

# ✅ Print Class Prediction Distribution
print("\n🔍 Class Predictions Distribution (Validation Set):")
for class_index, count in sorted(pred_counter.items()):
    print(f"{class_names[class_index]}: {count} predictions")

# ✅ Generate Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ✅ Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# ✅ Plot Confusion Matrix
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("CNN Confusion Matrix")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()


# input_folder = "crop/"
# output_folder = "crop_resized/"
# target_size = (224, 224)  # ✅ Optimal size for ResNet

# os.makedirs(output_folder, exist_ok=True)

# for class_name in os.listdir(input_folder):
#     class_folder = os.path.join(input_folder, class_name)
#     output_class_folder = os.path.join(output_folder, class_name)

#     if os.path.isdir(class_folder):
#         os.makedirs(output_class_folder, exist_ok=True)

#         for img_name in os.listdir(class_folder):
#             img_path = os.path.join(class_folder, img_name)
#             output_path = os.path.join(output_class_folder, img_name)

#             try:
#                 with Image.open(img_path) as img:
#                     img = img.resize(target_size, Image.LANCZOS)  # Best for quality
#                     sharpness = ImageEnhance.Sharpness(img)
#                     img = sharpness.enhance(2.0)  # Apply sharpening for details
#                     img.save(output_path)
#             except Exception as e:
#                 print(f"⚠️ Error processing {img_path}: {e}")

# print("✅ All images resized to 224x224 with sharpening applied!")

# output_folder = "crop_resized/"
# image_sizes = set()

# for class_name in os.listdir(output_folder):
#     class_folder = os.path.join(output_folder, class_name)
#     if os.path.isdir(class_folder):
#         for img_name in os.listdir(class_folder):
#             img_path = os.path.join(class_folder, img_name)
#             try:
#                 with Image.open(img_path) as img:
#                     image_sizes.add(img.size)
#             except Exception as e:
#                 print(f"⚠️ Error processing {img_path}: {e}")

# print("✅ Unique Image Sizes After Resizing:", image_sizes)















# # ✅ Load Class Names from dataset.yaml
# with open("dataset.yaml", "r") as f:
#     data_yaml = yaml.safe_load(f)

# class_names = data_yaml["names"]  # List of class names (['F16', 'F18', ..., 'B2'])
# class_to_idx = {name: idx for idx, name in enumerate(class_names)}  # Map class name → index



# # ✅ Load CNN Model
# class AircraftCNN(torch.nn.Module):
#     def __init__(self, num_classes=81):
#         super(AircraftCNN, self).__init__()
#         from torchvision import models
        
#         # Load ResNet18 with no pretrained weights explicitly
#         self.model = models.resnet18(weights=None)  # Change from pretrained=False

#         # Adjust the output layer to match the number of classes
#         self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

# cnn_model = AircraftCNN(num_classes=len(class_names))
# cnn_model.load_state_dict(torch.load("best_aircraft_model.pth"), strict=False)
# cnn_model.eval()
# print("✅ Model Loaded Successfully!")

# unique_preds = np.unique(cnn_model, return_counts=True)
# print(f"Unique Predicted Classes: {unique_preds}")  # Shows class distribution

# # ✅ Define Image Preprocessing (Same as training)
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Match CNN input size
#     transforms.ToTensor(),
# ])

# state_dict = torch.load("best_aircraft_model.pth")
# print("Model state_dict keys:", state_dict.keys())


# # ✅ Folder containing cropped test images
# test_folder = "crop/"

# # ✅ CNN Predictions
# y_true_cnn = []
# y_pred_cnn = []

# # ✅ Process images in the test set
# for class_name in os.listdir(test_folder):
#     class_folder = os.path.join(test_folder, class_name)
#     print(f"Checking test folder: {test_folder}")

#     if os.path.isdir(class_folder):
#         images = os.listdir(class_folder)
#         print(f"🔹 Checking class folder: {class_folder}, Found {len(images)} images")

#         for img_name in os.listdir(class_folder):
#             img_path = os.path.join(class_folder, img_name)
#             print(f"🔹 Processing image: {img_path}")

#             try:
#                 image = Image.open(img_path).convert("RGB")
#                 image = transform(image).unsqueeze(0)  # Add batch dimension

#                 with torch.no_grad():
#                     output = cnn_model(image)
#                     predicted_class = torch.argmax(output).item()

#                 # ✅ Convert actual class name to index for comparison
#                 y_true_cnn.append(class_to_idx[class_name])  # Actual class index
#                 y_pred_cnn.append(predicted_class)  # CNN predicted class index

#             except Exception as e:
#                 print(f"⚠️ Error processing {img_path}: {e}")

# # ✅ Compute Accuracy
# y_true_cnn = np.array(y_true_cnn)
# y_pred_cnn = np.array(y_pred_cnn)
# cnn_accuracy = np.mean(y_true_cnn == y_pred_cnn)

# print(f"✅ CNN Accuracy on Test Data: {cnn_accuracy:.4f}")

# # ✅ Generate Confusion Matrix
# cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)

# # ✅ Plot Confusion Matrix
# plt.figure(figsize=(14, 10))
# sns.heatmap(cm_cnn, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("CNN Confusion Matrix")
# plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=0)
# plt.show()

# # ✅ Print Classification Report
# print("Classification Report:")
# print(classification_report(y_true_cnn, y_pred_cnn, target_names=class_names))



# # Define dataset paths
# train_img_dir = '/home/loq/Documents/CourseWorkAML/src/MilitaryAircraftDataset/images/train'
# train_label_dir = '/home/loq/Documents/CourseWorkAML/src/MilitaryAircraftDataset/labels/train'
# val_img_dir = '/home/loq/Documents/CourseWorkAML/src/MilitaryAircraftDataset/images/val'
# val_label_dir = '/home/loq/Documents/CourseWorkAML/src/MilitaryAircraftDataset/labels/val'
# test_img_dir = '/home/loq/Documents/CourseWorkAML/src/MilitaryAircraftDataset/images/test'
# test_label_dir = '/home/loq/Documents/CourseWorkAML/src/MilitaryAircraftDataset/labels/test'

# # Create DataFrames
# train_image_files = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
# train_df = pd.DataFrame({
#     'image_path': [os.path.join(train_img_dir, img) for img in train_image_files]
# })
# train_df['label_path'] = train_df['image_path'].apply(
#     lambda x: os.path.join(train_label_dir, os.path.basename(x).replace('.jpg', '.txt'))
# )

# val_image_files = [f for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
# val_df = pd.DataFrame({
#     'image_path': [os.path.join(val_img_dir, img) for img in val_image_files]
# })
# val_df['label_path'] = val_df['image_path'].apply(
#     lambda x: os.path.join(val_label_dir, os.path.basename(x).replace('.jpg', '.txt'))
# )

# test_image_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
# test_df = pd.DataFrame({
#     'image_path': [os.path.join(test_img_dir, img) for img in test_image_files]
# })
# test_df['label_path'] = test_df['image_path'].apply(
#     lambda x: os.path.join(test_label_dir, os.path.basename(x).replace('.jpg', '.txt'))
# )

# # Verify label files (optional)
# for df, split in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
#     df['has_label'] = df['label_path'].apply(os.path.exists)
#     missing = df[~df['has_label']]
#     if not missing.empty:
#         print(f"Warning: {len(missing)} images in {split} have no labels:\n", missing)
#     df.drop(columns=['has_label'], inplace=True)

# print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

# # Define YOLOUtils instances and datasets
# train_utils = MyYOLOUtils(flip_prob=0.5, rotation_degrees=15, reshape_size=(640, 640))
# val_utils = MyYOLOUtils(flip_prob=0.0, rotation_degrees=0, reshape_size=(640, 640))

# train_dataset = CustomYOLODataset(train_df, train_img_dir, train_label_dir, train_utils)
# val_dataset = CustomYOLODataset(val_df, val_img_dir, val_label_dir, val_utils)
# test_dataset = CustomYOLODataset(test_df, test_img_dir, test_label_dir, val_utils)

# # Define DataLoader collate function
# def collate_fn(batch):
#     images, targets = zip(*batch)
#     images = torch.stack(images, 0)  # Stack images into a batch tensor
#     return images, targets  # Return images and list of targets

# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)

# # Define model and loss
# num_classes = 81  # Number of classes in your dataset
# model = YOLO('yolov8n.pt').model  # Access the underlying PyTorch model
# loss_fn = CustomYOLOLoss(num_classes=num_classes, yolo_utils=train_utils)

# # Move model and loss to device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# loss_fn.to(device)

# # Define optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()  # Set model to training mode
#     running_loss = 0.0

#     for images, targets in train_loader:
#         images = images.to(device)
#         targets = [t.to(device) for t in targets]

#         # Forward pass
#         optimizer.zero_grad()
#         outputs = model(images)  # Get model predictions
#         loss = loss_fn(outputs, targets)  # Compute loss
#         loss.backward()  # Backward pass
#         optimizer.step()  # Update weights

#         running_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

#     # Validation loop
#     model.eval()  # Set model to evaluation mode
#     val_loss = 0.0

#     with torch.no_grad():
#         for images, targets in val_loader:
#             images = images.to(device)
#             targets = [t.to(device) for t in targets]

#             outputs = model(images)
#             val_loss += loss_fn(outputs, targets).item()

#     print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

# # Save the trained model
# torch.save(model.state_dict(), 'custom_yolo.pt')

# # Step 4: Define DataFrames (data already split)
# train_img_dir = 'MilitaryAircraftDataset/images/train'
# train_label_dir = 'MilitaryAircraftDataset/labels/train'
# val_img_dir = 'MilitaryAircraftDataset/images/val'
# val_label_dir = 'MilitaryAircraftDataset/labels/val'
# test_img_dir = 'MilitaryAircraftDataset/images/test'
# test_label_dir = 'MilitaryAircraftDataset/labels/test'

# train_image_files = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
# train_df = pd.DataFrame({
#     'image_path': [os.path.join(train_img_dir, img) for img in train_image_files]
# })
# train_df['label_path'] = train_df['image_path'].apply(
#     lambda x: os.path.join(train_label_dir, os.path.basename(x).replace('.jpg', '.txt'))
# )

# val_image_files = [f for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
# val_df = pd.DataFrame({
#     'image_path': [os.path.join(val_img_dir, img) for img in val_image_files]
# })
# val_df['label_path'] = val_df['image_path'].apply(
#     lambda x: os.path.join(val_label_dir, os.path.basename(x).replace('.jpg', '.txt'))
# )

# test_image_files = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
# test_df = pd.DataFrame({
#     'image_path': [os.path.join(test_img_dir, img) for img in test_image_files]
# })
# test_df['label_path'] = test_df['image_path'].apply(
#     lambda x: os.path.join(test_label_dir, os.path.basename(x).replace('.jpg', '.txt'))
# )

# # Instantiate MyYOLOUtils and Datasets
# train_utils = MyYOLOUtils(flip_prob=0.5, rotation_degrees=15, reshape_size=(640, 640))
# val_utils = MyYOLOUtils(flip_prob=0.0, rotation_degrees=0, reshape_size=(640, 640))


# train_dataset = CustomYOLODataset(train_df, 'MilitaryAircraftDataset/images/train', 'MilitaryAircraftDataset/labels/train', train_utils)
# val_dataset = CustomYOLODataset(val_df, 'MilitaryAircraftDataset/images/val', 'MilitaryAircraftDataset/labels/val', val_utils)
# test_dataset = CustomYOLODataset(test_df, 'MilitaryAircraftDataset/images/test', 'MilitaryAircraftDataset/labels/test', val_utils)

# # Step 6: Define loss and model
# unique_classes = set()
# for label_path in train_df['label_path']:
#     with open(label_path, 'r') as f:
#         for line in f.readlines():
#             class_id = int(line.split()[0])
#             unique_classes.add(class_id)
# num_classes = len(unique_classes)

# loss_fn = CustomYOLOLoss(num_classes=num_classes, yolo_utils=train_utils)
# model = YOLO('yolov8n.pt')
# model.model.loss = loss_fn

# # Step 7: Optimizer and device setup
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# # Step 8: Import DataLoader and define loaders (moved to last)
# from torch.utils.data import DataLoader

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, 
#                          collate_fn=lambda x: tuple(zip(*x)))
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, 
#                        collate_fn=lambda x: tuple(zip(*x)))
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, 
#                         collate_fn=lambda x: tuple(zip(*x)))

# # Step 9: Training loop
# num_epochs = 50
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, targets in train_loader:
#         images = images.to(device)
#         targets = [t.to(device) for t in targets]
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = loss_fn(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for images, targets in val_loader:
#             images = images.to(device)
#             targets = [t.to(device) for t in targets]
#             outputs = model(images)
#             val_loss += loss_fn(outputs, targets).item()
#     print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

# torch.save(model.state_dict(), 'custom_yolo.pt')


# app = Flask(__name__, static_folder='static')

# # Load dataset.yaml
# with open("dataset.yaml", "r") as f:
#     data_yaml = yaml.safe_load(f)
# class_names = data_yaml["names"]

# # Load the trained YOLO model
# model = YOLO("runs/detect/train6/weights/best.pt")

# @app.route('/')
# def index():
#     return send_from_directory('static', 'index.html')

# @app.route('/detect', methods=['POST'])
# def detect():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400
    
#     file = request.files['image']
#     image = Image.open(file.stream)

#     # Run inference on the image
#     results = model(image, conf=0.3, iou=0.4)

#     # Convert image to NumPy array for visualization
#     image_np = np.array(image)

#     # Create a Matplotlib figure
#     plt.figure(figsize=(10, 6))
#     plt.imshow(image_np)
#     plt.axis("off")

#     # Draw bounding boxes and labels
#     for result in results:
#         for box in result.boxes:
#             cls = int(box.cls)
#             label = model.names[cls]
#             conf = box.conf.item()
#             x1, y1, x2, y2 = box.xyxy[0].tolist()

#             plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="red", linewidth=2, fill=False))
#             plt.text(x1, y1 - 5, f"{label} {conf:.2f}", color="red", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

#     # Save the result to a bytes buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=100)
#     buf.seek(0)
#     image_base64 = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close()

#     return jsonify({'image': image_base64})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
    
        
# import torch
# from ultralytics import YOLO
# import os
# from PIL import Image
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report, roc_curve, auc, f1_score
# import seaborn as sns
# from matplotlib.colors import LogNorm
# import yaml
# from sklearn.preprocessing import label_binarize
# import cv2

# with open("dataset.yaml", "r") as f:
#     data_yaml = yaml.safe_load(f)
# class_names = data_yaml["names"]

# # Load the trained model
# model = YOLO("runs/detect/train6/weights/best.pt")

# # Run inference on the image with a lower confidence threshold
# # results = model("F18_1.jpg", conf=0.01, iou=0.3)  # Lower conf

# # print("Index of F18:", class_names.index("F18"))

# # for result in results:
# #     for box in result.boxes:
# #         print(f"Detected: {class_names[int(box.cls)]} | Confidence: {box.conf.item():.4f} | Bbox: {box.xyxy.tolist()}")


# # for i, result in enumerate(results):
# #     result.show()  # ✅ Show detection
# #     save_path = f"output_F18_{i}.jpg"  # Save output
# #     # result.save(filename=save_path)
# #     print(f"Saved output to {save_path}")

# # Image path
# image_path = "MilitaryAircraftDataset/images/test/1e7dbc89eed2aee77edf62e861221be8.jpg"


# # Load the image
# image = Image.open(image_path)

# # Run inference on the image with a lower confidence threshold
# results = model(image_path, conf=0.3, iou=0.4)

# # Convert image to NumPy array for visualization
# image_np = np.array(image)

# # Create a Matplotlib figure 
# plt.figure(figsize=(10, 6))
# plt.imshow(image_np)
# plt.axis("off")  # Hide axis

# # Loop through detections and draw bounding boxes with class labels
# for result in results:
#     for box in result.boxes:
#         cls = int(box.cls)  # Class index
#         label = model.names[cls]  # Get class name
#         conf = box.conf.item()  # Confidence score
#         x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates

#         # Draw bounding box
#         plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="red", linewidth=2, fill=False))
        
#         # Add class label text
#         plt.text(x1, y1 - 5, f"{label} {conf:.2f}", color="red", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

# # Show the final image with detections
# plt.title("YOLO Detection Results")
# plt.show()











# # ✅ Load predictions
# with open("runs/detect/val/predictions.json", "r") as f:
#     predictions = json.load(f)

# # ✅ Load class names from dataset.yaml
# with open("dataset.yaml", "r") as f:
#     data_yaml = yaml.safe_load(f)
# class_names = data_yaml["names"]

# # ✅ Extract ground truth and predicted labels
# y_true = []
# y_scores = []

# for pred in predictions:
#     gt_label = pred["category_id"]  # Ground truth class
#     score = pred["score"]  # Confidence score of prediction
#     pred_label = pred["category_id"] if score > 0.5 else -1  # Threshold 0.5

#     y_true.append(gt_label)
#     y_scores.append(pred_label)

# # ✅ Convert to NumPy arrays
# y_true = np.array(y_true)
# y_scores = np.array(y_scores)

# # --------------------------------
# # **1️⃣ Precision-Recall Curve**
# # --------------------------------
# if len(np.unique(y_true)) > 1:  # Ensure multiple classes exist
#     precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)

#     plt.figure(figsize=(8, 6))
#     plt.plot(recall, precision, marker='.', label='PR Curve', color="blue")
#     plt.xlabel("Recall", fontsize=12)
#     plt.ylabel("Precision", fontsize=12)
#     plt.title("Precision-Recall Curve", fontsize=14)
#     plt.legend()
#     plt.grid()
#     plt.show()
# else:
#     print("⚠️ Skipping PR Curve: Only one class present in ground truth.")

# # --------------------------------
# # **2️⃣ Confusion Matrix (Top 40)**
# # --------------------------------
# cm = confusion_matrix(y_true, y_scores, labels=np.arange(len(class_names)))

# # ✅ Handle division errors (NaNs)
# with np.errstate(divide="ignore", invalid="ignore"):
#     class_accuracies = np.diag(cm) / cm.sum(axis=1)
#     class_accuracies = np.nan_to_num(class_accuracies)  # Replace NaNs with 0

# # ✅ Get top 40 most misclassified classes
# misclassified_sums = cm.sum(axis=1) - np.diag(cm)  # Total misclassified per class
# sorted_indices = np.argsort(misclassified_sums)[-40:]  # Get worst 40 classes

# # ✅ Create a confusion matrix for the top misclassified classes
# cm_top = cm[np.ix_(sorted_indices, sorted_indices)]

# # ✅ Get class names for these indices
# class_labels_sorted = [class_names[i] for i in sorted_indices]

# # ✅ Increase figure size for better visualization
# plt.figure(figsize=(14, 10))

# # ✅ Plot heatmap with proper class labels
# sns.heatmap(cm_top, annot=True, cmap="Reds", xticklabels=class_labels_sorted, yticklabels=class_labels_sorted, fmt="d")

# # ✅ Improve readability
# plt.xlabel("Predicted", fontsize=14)
# plt.ylabel("Actual", fontsize=14)
# plt.title("Confusion Matrix - Top 40 Misclassified Classes", fontsize=16)
# plt.xticks(rotation=45, ha="right")  # Rotate labels for better visibility
# plt.yticks(rotation=0)
# plt.show()

# # --------------------------------
# # **3️⃣ Per-Class Accuracy**
# # --------------------------------
# # ✅ Match `class_names` with `class_accuracies` length
# class_names = class_names[: len(class_accuracies)]

# plt.figure(figsize=(14, 6))
# plt.barh(class_names, class_accuracies, color="royalblue")
# plt.xlabel("Accuracy", fontsize=12)
# plt.ylabel("Class", fontsize=12)
# plt.title("Per-Class Accuracy", fontsize=14)
# plt.xlim([0, 1])
# plt.grid(axis="x")
# plt.show()

# # --------------------------------
# # **4️⃣ ROC Curve for Each Class**
# # --------------------------------
# if len(np.unique(y_true)) > 1:  # Ensure multiple classes exist
#     y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
#     y_scores_bin = label_binarize(y_scores, classes=np.arange(len(class_names)))

#     plt.figure(figsize=(10, 6))

#     # ✅ Plot ROC curve only for top 5 classes to keep it readable
#     for i in range(min(5, len(class_names))):  
#         fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores_bin[:, i])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

#     plt.xlabel("False Positive Rate", fontsize=12)
#     plt.ylabel("True Positive Rate", fontsize=12)
#     plt.title("ROC Curve for Top 5 Classes", fontsize=14)
#     plt.legend(loc="lower right")
#     plt.show()
# else:
#     print("⚠️ Skipping ROC Curve: Only one class present in ground truth.")

# # --------------------------------
# # **5️⃣ Per-Class F1 Score**
# # --------------------------------
# # ✅ Match `class_names` with `f1_scores` length
# if len(np.unique(y_true)) > 1:  # Ensure multiple classes exist
#     f1_scores = f1_score(y_true, y_scores, average=None)

#     # ✅ Ensure same number of labels in class_names and f1_scores
#     class_names = class_names[: len(f1_scores)]

#     plt.figure(figsize=(14, 6))
#     plt.barh(class_names, f1_scores, color="orange")
#     plt.xlabel("F1 Score", fontsize=12)
#     plt.ylabel("Class", fontsize=12)
#     plt.title("Per-Class F1 Score", fontsize=14)
#     plt.xlim([0, 1])
#     plt.grid(axis="x")
#     plt.show()
# else:
#     print("⚠️ Skipping F1 Score Plot: Only one class present in ground truth.")


# print("✅ YOLOv8 installed successfully!")
# print("CUDA Available:", torch.cuda.is_available())
# print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")



# predict_folder = "runs/detect/predict"
# images = [os.path.join(predict_folder, img) for img in os.listdir(predict_folder) if img.endswith(".jpg")]

# for img_path in images:
#     img = Image.open(img_path)
#     img.show()  # Opens the image

# model = YOLO("yolov8s.pt")

# model.train(
#     data="dataset.yaml",
#     epochs=8,
#     imgsz=640,
#     batch=8,  # Reduce if running out of VRAM
#     device="cuda"
# )
