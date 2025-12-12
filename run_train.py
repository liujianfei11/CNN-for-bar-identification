import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from dataset import GalaxyDataset
from model import build_model
from train import train_loop
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from plot import loss_curve
from plot import evaluate_and_confusion

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## 数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  ##50%概率水平翻转
        transforms.RandomRotation(20),  ##随机旋转，每次20deg
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  ##默认，不要改
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([  ##验证集不增强
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = GalaxyDataset('galaxy10_decals.h5', transform=train_transform)  ##提取image和ans(ans存进label列)列

    train_size = int(0.8 * len(full_dataset))  ##训练集、验证集、测试集规模8:1:1
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset ,test_dataset= random_split(full_dataset, [train_size, val_size, test_size])

    ## 应用transform
    train_dataset.dataset.set_transform(train_transform)
    val_dataset.dataset.set_transform(test_transform)
    test_dataset.dataset.set_transform(test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  ##batch32，防止过拟合
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)  ##验证集不需要打乱
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # model
    model = build_model().to(device)


    train_indices = train_dataset.indices  ## 训练集索引
    full_labels = full_dataset.labels
    train_labels = full_labels[train_indices]

    ## 逆频率权重
    barred = np.sum(train_labels == 1)
    unbarred = np.sum(train_labels == 0)
    weights = torch.tensor([len(train_labels) / unbarred, len(train_labels) / barred], dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)  ##定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4,
                                 weight_decay=5e-4)  ##定义优化器Adam，lr学习率，weight_decay是L2正则化强度

    ## model,image,label都应在GPU(cude:0)
    print('Model device:', next(model.parameters()).device)
    for images, labels in train_loader:
        print('Images device:', images.device)
        print('Labels device:', labels.device)
        break

    ## train
    train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve = train_loop(model, train_loader, val_loader,
                                                                                  criterion, optimizer, device,
                                                                                  epochs=6)

    ## loss curve and accuracy curve
    loss_curve(train_loss_curve, train_acc_curve, val_loss_curve, val_acc_curve)

    labels, preds, probs = evaluate_and_confusion(model, test_loader, device)

    torch.save(model.state_dict(), 'bar_classifier.pth')
    print('Saved')


if __name__ == '__main__':
    main()