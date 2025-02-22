import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# 自定义数据集
class XRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 数据预处理和增强
def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


# 加载数据
def load_data(control_dir, bacterial_dir, mycoplasma_dir, val_ratio=0.2):
    # 加载control图像 (label 0)
    control_images = [
        os.path.join(control_dir, img_name)
        for img_name in os.listdir(control_dir)
        if img_name.endswith(".jpg")
    ]

    # 加载bacterial图像 (label 1)
    bacterial_images = [
        os.path.join(bacterial_dir, img_name)
        for img_name in os.listdir(bacterial_dir)
        if img_name.endswith(".jpg")
    ]

    # 加载mycoplasma图像 (label 2)
    mycoplasma_images = [
        os.path.join(mycoplasma_dir, img_name)
        for img_name in os.listdir(mycoplasma_dir)
        if img_name.endswith(".jpg")
    ]

    # 在分割前打乱数据
    random.seed(42)
    random.shuffle(control_images)
    random.shuffle(bacterial_images)
    random.shuffle(mycoplasma_images)

    # 计算验证集大小
    n_control_val = int(len(control_images) * val_ratio)
    n_bacterial_val = int(len(bacterial_images) * val_ratio)
    n_mycoplasma_val = int(len(mycoplasma_images) * val_ratio)

    # 分割数据集
    train_paths = (
        control_images[n_control_val:]
        + bacterial_images[n_bacterial_val:]
        + mycoplasma_images[n_mycoplasma_val:]
    )

    train_labels = (
        [0] * (len(control_images) - n_control_val)
        + [1] * (len(bacterial_images) - n_bacterial_val)
        + [2] * (len(mycoplasma_images) - n_mycoplasma_val)
    )

    val_paths = (
        control_images[:n_control_val]
        + bacterial_images[:n_bacterial_val]
        + mycoplasma_images[:n_mycoplasma_val]
    )

    val_labels = [0] * n_control_val + [1] * n_bacterial_val + [2] * n_mycoplasma_val

    return train_paths, val_paths, train_labels, val_labels


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # 保存所有类别的概率

    # 计算各种指标
    f1 = f1_score(all_labels, all_preds, average="weighted")  # 使用weighted average
    # 对于多分类，我们使用macro平均的ROC AUC
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 计算每个类别的准确率
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    return {
        "f1_score": f1,
        "auc_score": auc,
        "confusion_matrix": conf_matrix,
        "class_accuracy": class_acc,
    }


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device="cuda"
):
    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # 评估训练集
        train_metrics = evaluate_model(model, train_loader, device)

        # 评估验证集
        val_metrics = evaluate_model(model, val_loader, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Training Loss: {running_loss / len(train_loader):.4f}")
        print(f"Training F1: {train_metrics['f1_score']: .4f}")
        print(f"Training AUC: {train_metrics['auc_score']: .4f}")
        print(f"Training Class Accuracy: {train_metrics['class_accuracy']}")
        print(f"Validation F1: {val_metrics['f1_score']: .4f}")
        print(f"Validation AUC: {val_metrics['auc_score']: .4f}")
        print(f"Validation Class Accuracy: {val_metrics['class_accuracy']}")
        print("Validation Confusion Matrix:")
        print(val_metrics["confusion_matrix"])
        print("--------------------")

        # 使用F1分数来保存最佳模型
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), "best_model.pth")

    return model


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    control_dir = "data/control"
    bacterial_dir = "data/bacterial"
    mycoplasma_dir = "data/mycoplasma"  # 新增mycoplasma目录

    X_train, X_val, y_train, y_val = load_data(
        control_dir, bacterial_dir, mycoplasma_dir, val_ratio=0.2
    )

    # 获取数据转换
    train_transform, val_transform = get_transforms()

    # 创建数据集
    train_dataset = XRayDataset(X_train, y_train, train_transform)
    val_dataset = XRayDataset(X_val, y_val, val_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 加载预训练的MobileNet v2
    model = models.mobilenet_v2(pretrained=True)

    # 冻结大部分层
    for param in model.parameters():
        param.requires_grad = False

    # 修改最后的分类器层
    model.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 3),  # 修改为3个类别
    )

    # 只训练最后几层
    for param in model.features[-3:].parameters():  # 训练最后3个卷积块
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {"params": model.features[-3:].parameters()},
            {"params": model.classifier.parameters(), "lr": 0.001},
        ],
        lr=0.0001,
    )

    # 训练模型
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=50,
        device=device,
    )


if __name__ == "__main__":
    main()
