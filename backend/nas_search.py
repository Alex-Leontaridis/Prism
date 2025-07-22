import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time
import json
import csv
import os
from nas_strategies import get_search_space, get_search_strategy, ConstraintParser
import sqlite3
import mlflow
import wandb

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

POP_SIZE = 2
GENERATIONS = 2
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
        self.skip_connections = {}
        in_channels = 1
        flatten = False
        layer_outputs = []  # For skip/residual
        for idx, layer in enumerate(arch['layers']):
            if layer['type'] == 'Conv2D':
                self.features.append(nn.Conv2d(in_channels, layer['filters'], layer['kernel'], padding=1))
                in_channels = layer['filters']
                flatten = True
            elif layer['type'] == 'BatchNorm2D':
                self.features.append(nn.BatchNorm2d(in_channels))
            elif layer['type'] == 'LayerNorm':
                self.features.append(nn.LayerNorm([in_channels, 28, 28]))
            elif layer['type'] == 'MaxPool2D':
                self.features.append(nn.MaxPool2d(layer['pool_size']))
                flatten = True
            elif layer['type'] == 'Dropout':
                self.features.append(nn.Dropout(layer['rate']))
            elif layer['type'] == 'Residual':
                # Mark residual connection for forward
                self.skip_connections[len(self.features)] = layer['from']
            elif layer['type'] == 'Skip':
                self.skip_connections[len(self.features)] = layer['from']
            elif layer['type'] == 'Dense':
                break
        self.features.append(nn.Flatten())
        dummy = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            x = dummy
            for i, layer in enumerate(self.features):
                prev_x = x
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    x = F.relu(x)
                if i in self.skip_connections:
                    from_idx = self.skip_connections[i]
                    if from_idx < len(self.features):
                        x = x + dummy if from_idx == 0 else layer_outputs[from_idx]
                layer_outputs.append(x)
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
        layer_outputs = [x]
        for i, layer in enumerate(self.features):
            prev_x = x
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = F.relu(x)
            if i in self.skip_connections:
                from_idx = self.skip_connections[i]
                if from_idx < len(layer_outputs):
                    x = x + layer_outputs[from_idx]
            layer_outputs.append(x)
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if isinstance(layer, nn.Linear) and i < len(self.classifier) - 1:
                x = F.relu(x)
        return x

# --- Training & Evaluation ---
def train_and_eval(arch, epochs=EPOCHS, patience=3, max_seconds=None):
    model = NASNet(arch).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
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
        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= total
        val_acc = correct / total
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if patience and epochs_no_improve >= patience:
            break
        if max_seconds and (time.time() - start) > max_seconds:
            break
    train_time = time.time() - start
    num_params = sum(p.numel() for p in model.parameters())
    return {'val_acc': val_acc, 'num_params': num_params, 'train_time': train_time, 'val_loss': best_val_loss}

class ExperimentLogger:
    def __init__(self, method='json', project_name='prism_nas', run_name=None, db_path='nas_experiments.db', json_path='nas_experiments.json'):
        self.method = method
        self.project_name = project_name
        self.run_name = run_name
        self.db_path = db_path
        self.json_path = json_path
        self.json_data = []
        if method == 'sqlite':
            self.conn = sqlite3.connect(db_path)
            self._init_sqlite()
        elif method == 'mlflow':
            mlflow.set_experiment(project_name)
            self.mlflow_run = mlflow.start_run(run_name=run_name)
        elif method == 'wandb':
            wandb.init(project=project_name, name=run_name)
    def _init_sqlite(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arch TEXT,
            metrics TEXT,
            config TEXT
        )''')
        self.conn.commit()
    def log(self, arch, metrics, config=None):
        if self.method == 'mlflow':
            mlflow.log_params({f'arch_{i}': str(l) for i, l in enumerate(arch['layers'])})
            mlflow.log_metrics(metrics)
            if config:
                mlflow.log_params(config)
        elif self.method == 'wandb':
            wandb.log({**metrics, 'arch': str(arch), **(config or {})})
        elif self.method == 'sqlite':
            c = self.conn.cursor()
            c.execute('INSERT INTO experiments (arch, metrics, config) VALUES (?, ?, ?)',
                      (json.dumps(arch), json.dumps(metrics), json.dumps(config or {})))
            self.conn.commit()
        else:  # json
            self.json_data.append({'arch': arch, 'metrics': metrics, 'config': config})
            with open(self.json_path, 'w') as f:
                json.dump(self.json_data, f, indent=2)
    def close(self):
        if self.method == 'mlflow':
            mlflow.end_run()
        elif self.method == 'wandb':
            wandb.finish()
        elif self.method == 'sqlite':
            self.conn.close()

# --- Pluggable NAS Search Loop ---
def nas_search(strategy_name='random', constraints=None, prior_arch_path=None, log_method='json', progress_callback=None):
    search_space = get_search_space()
    constraint_parser = ConstraintParser(constraints)
    prior_arch = None
    if prior_arch_path and os.path.exists(prior_arch_path):
        with open(prior_arch_path, 'r') as f:
            prior_arch = json.load(f)
    strategy = get_search_strategy(
        strategy_name,
        search_space,
        pop_size=POP_SIZE,
        top_k=TOP_K,
        constraint_parser=constraint_parser
    )
    # If prior_arch is provided, seed the initial population
    if prior_arch and hasattr(strategy, 'population'):
        strategy.population[0] = prior_arch
    history = []
    total_iters = POP_SIZE * GENERATIONS
    logger = ExperimentLogger(method=log_method)
    for i in range(total_iters):
        arch = strategy.generate_next_architecture()
        metrics = train_and_eval(arch)
        if not constraint_parser.check(arch, metrics):
            print(f"WARNING: Constraint violation at iter {i+1} (metrics: {metrics})")
        strategy.update_with_result(arch, metrics)
        score = metrics['val_acc'] - 1e-6 * metrics['num_params'] - constraint_parser.penalty(arch, metrics)
        history.append({'arch': arch, 'metrics': metrics, 'score': score})
        logger.log(arch, metrics, config={'strategy': strategy_name, 'constraints': constraints})
        print(f"Iter {i+1}/{total_iters} | Acc: {metrics['val_acc']:.4f} | Params: {metrics['num_params']} | Time: {metrics['train_time']:.1f}s | Penalty: {constraint_parser.penalty(arch, metrics):.4f}")
        if progress_callback:
            progress_callback({
                'iteration': i + 1,
                'total_trials': total_iters,
                'accuracy': metrics.get('val_acc'),
                'loss': metrics.get('val_loss'),
                'status': f"Running trial {i+1}/{total_iters}"
            })
    if progress_callback:
        progress_callback({
            'iteration': total_iters,
            'total_trials': total_iters,
            'status': 'completed'
        })
    logger.close()
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
    parser.add_argument('--max_params', type=int, default=None, help='Maximum allowed number of parameters')
    parser.add_argument('--min_accuracy', type=float, default=None, help='Minimum required validation accuracy')
    parser.add_argument('--prior_arch', type=str, default=None, help='Path to prior architecture JSON file')
    parser.add_argument('--log_method', type=str, default='json', help='Experiment logger: json, sqlite, mlflow, wandb')
    args = parser.parse_args()
    constraints = {}
    if args.max_params is not None:
        constraints['max_params'] = args.max_params
    if args.min_accuracy is not None:
        constraints['min_accuracy'] = args.min_accuracy
    nas_search(strategy_name=args.strategy, constraints=constraints, prior_arch_path=args.prior_arch, log_method=args.log_method) 