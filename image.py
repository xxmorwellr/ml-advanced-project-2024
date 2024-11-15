import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet # pip install efficientnet-pytorch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import pickle

# Load labels from CSV
train_df = pd.read_csv("train_subset.csv")
unique_labels = pd.unique(train_df['label_group'])
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
num_classes = len(label_mapping)

# Define a custom dataset to load images and labels
class ImageDataset(Dataset):
    def __init__(self, csv_file, image_dir, label_mapping, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_labels = dict(zip(self.data['image'], self.data['label_group'].map(label_mapping)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        posting_id = self.data.iloc[idx]['posting_id']
        img_name = self.data.iloc[idx]['image']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.image_labels[img_name]
        return image, label, posting_id  # Return posting_id for identification

# Data pre-processing and loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

csv_file = 'train_subset.csv'
image_dir = 'train_images'
print("Loading data...")
dataset = ImageDataset(csv_file=csv_file, image_dir=image_dir, label_mapping=label_mapping, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ArcFace Loss Layer
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, features, labels):
        cos_theta = torch.matmul(features, self.weight.t())
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        target_logits = torch.cos(theta + self.margin)
        
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * target_logits + (1.0 - one_hot) * cos_theta
        logits *= self.scale
        return logits

# EfficientNet with ArcFace for Feature Extraction
class EfficientNetWithArcFace(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet-b0'): 
        super(EfficientNetWithArcFace, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(model_name)
        self.feature_dim = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()  # Remove final layer
        self.arcface = ArcFaceLoss(in_features=self.feature_dim, out_features=num_classes)

    def forward(self, x, labels=None):
        features = self.efficientnet(x)
        if labels is not None:
            return self.arcface(features, labels)
        return features

# Initialize model and device
model_name = 'efficientnet-b0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Initialize EfficientNet model...")
model = EfficientNetWithArcFace(num_classes=num_classes, model_name=model_name).to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 10

# Training loop
print("Start training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images, labels)  # ArcFace returns logits
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")
print("Training completed!")

model.eval()  # Set model to evaluation mode
image_features = []
image_labels = []
posting_ids = []

with torch.no_grad():
    for images, labels, post_ids in dataloader:
        images = images.to(device)
        features = model(images)  # Extract ArcFace-enhanced features
        image_features.append(features.cpu().numpy())
        image_labels.extend(labels.numpy())
        posting_ids.extend(post_ids)

# Combine all extracted features
features_matrix = np.vstack(image_features)
labels_vector = np.array(image_labels)

# # Save for further cocatenation
# torch.save(features_matrix, 'image_features.pt')
# torch.save(posting_ids, 'posting_ids.pt')

# Use KNN for finding similar products
knn_model = NearestNeighbors(n_neighbors=2, metric="cosine")
knn_model.fit(features_matrix)

# Find similar products for each image in the dataset
similar_products = {}
for i, feature in enumerate(features_matrix):
    distances, indices = knn_model.kneighbors([feature])
    similar_indices = indices.flatten()
    similar_product_ids = [posting_ids[idx] for idx in similar_indices if idx != i]  # exclude itself
    similar_products[posting_ids[i]] = similar_product_ids

# Save
with open('knn_image_matches.pkl', 'wb') as f:
    pickle.dump(similar_products, f)

print("KNN image matches has been saved!")