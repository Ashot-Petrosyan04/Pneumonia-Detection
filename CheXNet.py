import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score

BATCH_SIZE = 16
LR = 0.001
EPOCHS = 10

class ChestXRayDataset(Dataset):
    def __init__(self, phase='train'):
        self.image_paths = []
        self.labels = []
        self.phase = phase
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        with open('sample_labels.csv', 'r') as f:
            next(f)
            for line in f:
                items = line.strip().split(',')
                img_path = os.path.join('sample/images/', items[0].strip('"'))
                pathologies = items[1].strip('"').split('|')
                label = 1.0 if 'Pneumonia' in pathologies else 0.0
                self.image_paths.append(img_path)
                self.labels.append(label)

        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                self.normalize
            ])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(self.labels[index], dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)

class DenseNet121(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.densenet(x)

def main():
    full_dataset = ChestXRayDataset(phase='train')
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    pos_count = sum(full_dataset.labels[i] for i in train_dataset.indices)
    neg_count = len(train_dataset) - pos_count
    pos_weight = torch.tensor([neg_count/pos_count])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DenseNet121()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            print(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs.view(-1), labels)
                val_loss += loss.item() * images.size(0)
        
        avg_val_loss = val_loss / len(val_dataset)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss/len(train_dataset):.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}\n')

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = torch.sigmoid(model(images))
            predicted = (outputs >= 0.5).float().view(-1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    print(f'Test F1 Score: {f1:.4f}')

if __name__ == '__main__':
    main()
