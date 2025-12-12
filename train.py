import torch
from tqdm import tqdm

def train_loop(model, train_loader, test_loader, criterion, optimizer, device, epochs=5):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, epochs+1):
        print(f'\nEpoch {epoch}/{epochs}')

        model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc='Training', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = train_correct / total


        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc='Validating', leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    return train_losses, train_accs, val_losses, val_accs



