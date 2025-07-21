import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import json
import csv
from nas_strategies import get_search_strategy

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)

# Search space
def get_search_space():
    return {
        'LAYER_TYPES': ['Conv2D', 'MaxPool2D', 'Dense', 'Dropout'],
        'CONV_FILTERS': [16, 32, 64],
        'KERNEL_SIZES': [3, 5],
        'POOL_SIZES': [2],
        'DENSE_UNITS': [64, 128, 256],
        'DROPOUT_RATES': [0.2, 0.3, 0.5],
    }

POP_SIZE = 6
GENERATIONS = 10
TOP_K = 3
EPOCHS = 3

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('data/MNIST', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data/MNIST', train=False, download=True, transform=transform)

train_len = int(0.8 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_data, val_data = random_split(train_dataset, [train_len, val_len])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

# --- Model Builder ---
class NASNet(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.features = nn.ModuleList()
        self.classifier = nn.ModuleList()
        in_channels = 1
        flatten = False
        for layer in arch['layers']:
            if layer['type'] == 'Conv2D':
                self.features.append(nn.Conv2d(in_channels, layer['filters'], layer['kernel'], padding=1))
                in_channels = layer['filters']
                flatten = True
            elif layer['type'] == 'MaxPool2D':
                self.features.append(nn.MaxPool2d(layer['pool_size']))
                flatten = True
            elif layer['type'] == 'Dropout':
                self.features.append(nn.Dropout(layer['rate']))
            elif layer['type'] == 'Dense':
                break
        self.features.append(nn.Flatten())
        dummy = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            x = dummy
            for layer in self.features:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    x = F.relu(x)
            in_features = x.shape[1]
        dense_started = False
        for layer in arch['layers']:
            if layer['type'] == 'Dense':
                self.classifier.append(nn.Linear(in_features, layer['units']))
                in_features = layer['units']
                dense_started = True
            elif layer['type'] == 'Dropout' and dense_started:
                self.classifier.append(nn.Dropout(layer['rate']))
        if len(self.classifier) == 0:
            self.classifier.append(nn.Linear(28*28, 10))
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = F.relu(x)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if isinstance(layer, nn.Linear) and i < len(self.classifier) - 1:
                x = F.relu(x)
        return x

# --- Training & Evaluation ---
def train_and_eval(arch, epochs=EPOCHS):
    model = NASNet(arch).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total
    num_params = sum(p.numel() for p in model.parameters())
    return {'val_acc': val_acc, 'num_params': num_params, 'train_time': train_time}

# --- Pluggable NAS Search Loop ---
def nas_search(strategy_name='random'):
    search_space = get_search_space()
    strategy = get_search_strategy(
        strategy_name,
        search_space,
        pop_size=POP_SIZE,
        top_k=TOP_K
    )
    history = []
    total_iters = POP_SIZE * GENERATIONS
    for i in range(total_iters):
        arch = strategy.generate_next_architecture()
        metrics = train_and_eval(arch)
        strategy.update_with_result(arch, metrics)
        score = metrics['val_acc'] - 1e-6 * metrics['num_params']
        history.append({'arch': arch, 'metrics': metrics, 'score': score})
        print(f"Iter {i+1}/{total_iters} | Acc: {metrics['val_acc']:.4f} | Params: {metrics['num_params']} | Time: {metrics['train_time']:.1f}s")
    # Save history
    with open('nas_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open('nas_history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['arch', 'val_acc', 'num_params', 'train_time', 'score'])
        for h in history:
            writer.writerow([json.dumps(h['arch']), h['metrics']['val_acc'], h['metrics']['num_params'], h['metrics']['train_time'], h['score']])
    print('NAS search complete. Results saved to nas_history.json and nas_history.csv')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='random', help='Search strategy: random, evolutionary, bayesian, etc.')
    args = parser.parse_args()
    nas_search(strategy_name=args.strategy) 