import torch.nn as nn
import torch

def train_model(model, dataloader, lr, num_epochs , device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-4)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        avg_loss = running_loss / num_batches
        print(f"    Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    return model.state_dict()
    
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct / total
