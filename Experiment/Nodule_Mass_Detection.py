import timm
import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

BATCH_SIZE = 16
LR = 0.001
EPOCHS = 10

class ChestXRayDataset(Dataset):
    def __init__(self, phase='train'):
        self.image_paths = []
        self.labels = []
        self.phase = phase
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        with open('balanced_data.csv', 'r') as f:
            next(f)
            for line in f:
                items = line.strip().split(',')
                pathologies = items[1].strip('"').split('|')
                target_pathologies = {'Nodule', 'Mass'}
                label = 1.0 if any(p in target_pathologies for p in pathologies) else 0.0
                img_path = os.path.join('no_finding/' if label == 0.0 else 'mass_nodule_finding/', items[0].strip('"'))
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

        for param in self.densenet.parameters():
            param.requires_grad = False

        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.densenet(x)
    
class VGG16Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.vgg(x)

class InceptionNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V3, aux_logits=False)
        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.inception(x)

class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet(x)

class XceptionNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception = timm.create_model('xception', pretrained=True)

        for param in self.xception.parameters():
            param.requires_grad = False
        
        num_features = self.xception.get_classifier().in_features
        self.xception.reset_classifier(1)
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.xception.forward_features(x)
        return self.classifier(features)

class MobileNetV2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)

        for param in self.mobilenet.features.parameters():
            param.requires_grad = False

        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.mobilenet(x)

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, pos_weight):
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"\n----- Training {model_name} -----")
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
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss/len(train_loader.dataset):.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}\n')
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = torch.sigmoid(model(images))
            probs = outputs.view(-1).cpu().numpy()
            predicted = (probs >= 0.5).astype(float)
            all_preds.extend(predicted)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    print(f'----- {model_name} Test Metrics -----')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {auc:.4f}\n')
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"{model_name}_roc_curve.png")
    plt.close()
    
    torch.save(model.state_dict(), f"{model_name}_final_model.pth")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

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

    models = {
        "DenseNet121": DenseNet121(),
        # "VGG16": VGG16Model(),
        # "InceptionNet": InceptionNetModel(),
        # "ResNet": ResNetModel(),
        #  "MobileNetV2": MobileNetV2Model(),
        #  "XceptionNet": XceptionNetModel()
    }
    
    results = {}
    for name, model in models.items():
        results[name] = train_and_evaluate(model, name, train_loader, val_loader, test_loader, pos_weight)
    
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print("----- Overall Results -----")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")
    print(f"\nBest Model based on F1 Score: {best_model[0]} with F1 Score = {best_model[1]['f1']:.4f}")

if __name__ == '__main__':
    main()
