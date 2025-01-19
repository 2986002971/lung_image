import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
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
def load_data(control_dir, bacterial_dir, val_ratio=0.2):
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

    # 在分割前打乱数据
    random.seed(42)  # 设置随机种子以保证可重复性
    random.shuffle(control_images)
    random.shuffle(bacterial_images)

    # 计算验证集大小
    n_control_val = int(len(control_images) * val_ratio)
    n_bacterial_val = int(len(bacterial_images) * val_ratio)

    # 分割数据集
    train_paths = control_images[n_control_val:] + bacterial_images[n_bacterial_val:]
    train_labels = [0] * (len(control_images) - n_control_val) + [1] * (
        len(bacterial_images) - n_bacterial_val
    )

    val_paths = control_images[:n_control_val] + bacterial_images[:n_bacterial_val]
    val_labels = [0] * n_control_val + [1] * n_bacterial_val

    return train_paths, val_paths, train_labels, val_labels


# 模型训练函数
def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device="cuda"
):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Training Loss: {running_loss / len(train_loader):.4f}")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")
        print("--------------------")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    return model


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    control_dir = "data/control"  # 替换为您的control图像目录
    bacterial_dir = "data/bacterial"  # 替换为您的bacterial图像目录

    X_train, X_val, y_train, y_val = load_data(
        control_dir, bacterial_dir, val_ratio=0.2
    )

    # 获取数据转换
    train_transform, val_transform = get_transforms()

    # 创建数据集
    train_dataset = XRayDataset(X_train, y_train, train_transform)
    val_dataset = XRayDataset(X_val, y_val, val_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 加载预训练的ResNet50
    model = models.resnet50(pretrained=True)

    # 冻结大部分层
    for param in model.parameters():
        param.requires_grad = False

    # 修改最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),  # 2个类别
    )

    # 只训练最后几层
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {"params": model.layer4.parameters()},
            {"params": model.fc.parameters(), "lr": 0.001},
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
