import torch
from nas_strategies import get_search_strategy, get_search_space
from nas_search import NASNet, train_and_eval
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

print('DEBUG: Script started')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data setup (reuse from nas_search.py)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('data/MNIST', train=True, download=True, transform=transform)
train_len = int(0.8 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_data, val_data = random_split(train_dataset, [train_len, val_len])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

print('DEBUG: Data loaded')

# Patch train_and_eval to use local loaders
def train_and_eval_local(arch, epochs=2):
    model = NASNet(arch).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
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
    return {'val_acc': val_acc, 'num_params': num_params}


def test_strategy(strategy_name, iters=5):
    print(f'\nDEBUG: Testing strategy: {strategy_name}')
    search_space = get_search_space()
    strategy = get_search_strategy(strategy_name, search_space, pop_size=3, top_k=1)
    for i in range(iters):
        print(f'DEBUG: Iter {i+1} for {strategy_name}')
        arch = strategy.generate_next_architecture()
        metrics = train_and_eval_local(arch, epochs=1)
        strategy.update_with_result(arch, metrics)
        print(f"Iter {i+1}: Acc={metrics['val_acc']:.4f}, Params={metrics['num_params']}, Arch={arch['layers']}")

if __name__ == '__main__':
    print('DEBUG: Starting main test loop')
    for strat in ['random', 'evolutionary']:
        test_strategy(strat, iters=5)
    print('DEBUG: All tests complete') 