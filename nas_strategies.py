import random
import json

# --- Base Strategy ---
class SearchStrategy:
    def __init__(self, search_space, **kwargs):
        self.search_space = search_space
    def generate_next_architecture(self):
        raise NotImplementedError
    def update_with_result(self, arch, metrics):
        pass

# --- Random Search ---
class RandomSearchStrategy(SearchStrategy):
    def generate_next_architecture(self):
        return random_architecture(self.search_space)

# --- Evolutionary Search ---
class EvolutionarySearchStrategy(SearchStrategy):
    def __init__(self, search_space, pop_size=6, top_k=3, **kwargs):
        super().__init__(search_space)
        self.pop_size = pop_size
        self.top_k = top_k
        self.population = [random_architecture(self.search_space) for _ in range(self.pop_size)]
        self.scores = []
        self.generation = 0
    def generate_next_architecture(self):
        if self.generation == 0 or len(self.scores) < self.pop_size:
            # Initial population
            return self.population[len(self.scores)]
        # Mutate top-K
        top_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)[:self.top_k]
        parent = self.population[random.choice(top_indices)]
        child = mutate_arch(parent, self.search_space)
        return child
    def update_with_result(self, arch, metrics):
        score = score_arch(metrics)
        self.scores.append(score)
        if len(self.scores) == self.pop_size:
            # New generation
            top_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)[:self.top_k]
            new_population = [self.population[i] for i in top_indices]
            while len(new_population) < self.pop_size:
                parent = new_population[random.randint(0, self.top_k-1)]
                child = mutate_arch(parent, self.search_space)
                new_population.append(child)
            self.population = new_population
            self.scores = []
            self.generation += 1

# --- Bayesian Search (Stub) ---
class BayesianSearchStrategy(SearchStrategy):
    def generate_next_architecture(self):
        raise NotImplementedError('Bayesian search not implemented yet.')

# --- Reinforcement Search (Stub) ---
class ReinforcementSearchStrategy(SearchStrategy):
    def generate_next_architecture(self):
        raise NotImplementedError('Reinforcement search not implemented yet.')

# --- Grid/Manual Search (Stub) ---
class GridSearchStrategy(SearchStrategy):
    def generate_next_architecture(self):
        raise NotImplementedError('Grid/manual search not implemented yet.')

# --- Factory ---
def get_search_strategy(name, search_space, **kwargs):
    name = name.lower()
    if name == 'random':
        return RandomSearchStrategy(search_space, **kwargs)
    elif name == 'evolutionary':
        return EvolutionarySearchStrategy(search_space, **kwargs)
    elif name == 'bayesian':
        return BayesianSearchStrategy(search_space, **kwargs)
    elif name == 'reinforcement':
        return ReinforcementSearchStrategy(search_space, **kwargs)
    elif name == 'grid':
        return GridSearchStrategy(search_space, **kwargs)
    else:
        raise ValueError(f'Unknown strategy: {name}')

def get_search_space():
    return {
        'LAYER_TYPES': ['Conv2D', 'MaxPool2D', 'Dense', 'Dropout'],
        'CONV_FILTERS': [16, 32, 64],
        'KERNEL_SIZES': [3, 5],
        'POOL_SIZES': [2],
        'DENSE_UNITS': [64, 128, 256],
        'DROPOUT_RATES': [0.2, 0.3, 0.5],
    }

# --- Utility: random_architecture, mutate_arch, score_arch ---
def random_architecture(search_space):
    LAYER_TYPES = search_space['LAYER_TYPES']
    CONV_FILTERS = search_space['CONV_FILTERS']
    KERNEL_SIZES = search_space['KERNEL_SIZES']
    POOL_SIZES = search_space['POOL_SIZES']
    DENSE_UNITS = search_space['DENSE_UNITS']
    DROPOUT_RATES = search_space['DROPOUT_RATES']
    layers = []
    num_layers = random.randint(2, 5)
    for i in range(num_layers):
        layer_type = random.choice(LAYER_TYPES)
        if layer_type == 'Conv2D':
            layers.append({'type': 'Conv2D', 'filters': random.choice(CONV_FILTERS), 'kernel': random.choice(KERNEL_SIZES)})
        elif layer_type == 'MaxPool2D':
            layers.append({'type': 'MaxPool2D', 'pool_size': random.choice(POOL_SIZES)})
        elif layer_type == 'Dense':
            layers.append({'type': 'Dense', 'units': random.choice(DENSE_UNITS)})
        elif layer_type == 'Dropout':
            layers.append({'type': 'Dropout', 'rate': random.choice(DROPOUT_RATES)})
    layers.append({'type': 'Dense', 'units': 10})
    return {'layers': layers}

def mutate_arch(arch, search_space):
    new_arch = json.loads(json.dumps(arch))
    layers = new_arch['layers']
    LAYER_TYPES = search_space['LAYER_TYPES']
    CONV_FILTERS = search_space['CONV_FILTERS']
    KERNEL_SIZES = search_space['KERNEL_SIZES']
    POOL_SIZES = search_space['POOL_SIZES']
    DENSE_UNITS = search_space['DENSE_UNITS']
    DROPOUT_RATES = search_space['DROPOUT_RATES']
    op = random.choice(['add', 'remove', 'change'])
    if op == 'add' and len(layers) < 8:
        idx = random.randint(0, len(layers)-2)
        layer_type = random.choice(LAYER_TYPES)
        if layer_type == 'Conv2D':
            new_layer = {'type': 'Conv2D', 'filters': random.choice(CONV_FILTERS), 'kernel': random.choice(KERNEL_SIZES)}
        elif layer_type == 'MaxPool2D':
            new_layer = {'type': 'MaxPool2D', 'pool_size': random.choice(POOL_SIZES)}
        elif layer_type == 'Dense':
            new_layer = {'type': 'Dense', 'units': random.choice(DENSE_UNITS)}
        elif layer_type == 'Dropout':
            new_layer = {'type': 'Dropout', 'rate': random.choice(DROPOUT_RATES)}
        layers.insert(idx, new_layer)
    elif op == 'remove' and len(layers) > 2:
        idx = random.randint(0, len(layers)-2)
        layers.pop(idx)
    elif op == 'change':
        idx = random.randint(0, len(layers)-2)
        layer = layers[idx]
        if layer['type'] == 'Conv2D':
            layer['filters'] = random.choice(CONV_FILTERS)
            layer['kernel'] = random.choice(KERNEL_SIZES)
        elif layer['type'] == 'MaxPool2D':
            layer['pool_size'] = random.choice(POOL_SIZES)
        elif layer['type'] == 'Dense':
            layer['units'] = random.choice(DENSE_UNITS)
        elif layer['type'] == 'Dropout':
            layer['rate'] = random.choice(DROPOUT_RATES)
    return new_arch

def score_arch(metrics):
    return metrics['val_acc'] - 1e-6 * metrics['num_params'] 