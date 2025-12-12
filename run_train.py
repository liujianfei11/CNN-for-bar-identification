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

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = GalaxyDataset('galaxy10_decals.h5', transform=train_transform) 

    train_size = int(0.8 * len(full_dataset)) 
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset ,test_dataset= random_split(full_dataset, [train_size, val_size, test_size])

    ## 应用transform
    train_dataset.dataset.set_transform(train_transform)
    val_dataset.dataset.set_transform(test_transform)
    test_dataset.dataset.set_transform(test_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0) 
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0) 
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = build_model().to(device)


    train_indices = train_dataset.indices 
    full_labels = full_dataset.labels
    train_labels = full_labels[train_indices]


    barred = np.sum(train_labels == 1)
    unbarred = np.sum(train_labels == 0)
    weights = torch.tensor([len(train_labels) / unbarred, len(train_labels) / barred], dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights) 
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4,
                                 weight_decay=5e-4) 

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
