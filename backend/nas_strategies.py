import random
import json
import optuna
import numpy as np
import skopt
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args

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
    def __init__(self, search_space, use_nasbench=False, nasbench_version='101', **kwargs):
        super().__init__(search_space)
        self.use_nasbench = use_nasbench
        self.nasbench_version = nasbench_version
        if use_nasbench:
            if nasbench_version == '101':
                self.nasbench = NASBench101Backend()
            else:
                self.nasbench = NASBench201Backend()
    def generate_next_architecture(self):
        return random_architecture(self.search_space)
    def update_with_result(self, arch, metrics):
        pass
    def evaluate(self, arch):
        if self.use_nasbench:
            return self.nasbench.evaluate(arch)
        else:
            # Placeholder for real training/eval
            return {'val_acc': 0.8, 'num_params': 10000}

# --- Evolutionary Search ---
class EvolutionarySearchStrategy(SearchStrategy):
    def __init__(self, search_space, pop_size=6, top_k=3, constraint_parser=None, **kwargs):
        super().__init__(search_space)
        self.pop_size = pop_size
        self.top_k = top_k
        self.constraint_parser = constraint_parser
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
        score = score_arch(metrics, self.constraint_parser, arch)
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
    def __init__(self, search_space, n_calls=20, **kwargs):
        super().__init__(search_space)
        self.n_calls = n_calls
        self.results = []
        # Define the search space for skopt
        self.space = [
            Integer(2, 5, name='num_layers'),
            Categorical(search_space['LAYER_TYPES'], name='layer_type'),
            Categorical(search_space['CONV_FILTERS'], name='filters'),
            Categorical(search_space['KERNEL_SIZES'], name='kernel'),
            Categorical(search_space['POOL_SIZES'], name='pool_size'),
            Categorical(search_space['DENSE_UNITS'], name='units'),
            Categorical(search_space['DROPOUT_RATES'], name='dropout_rate'),
            Categorical(search_space['LEARNING_RATES'], name='learning_rate'),
            Categorical(search_space['OPTIMIZERS'], name='optimizer'),
            Categorical(search_space['BATCH_SIZES'], name='batch_size'),
        ]
        self.optimizer = skopt.Optimizer(self.space)
        self.asked = 0
        self.arch_queue = []

    def generate_next_architecture(self):
        if self.asked >= self.n_calls:
            return None
        params = self.optimizer.ask()
        # Map params to architecture
        num_layers = params[0]
        layers = []
        for i in range(num_layers):
            layer_type = params[1]
            if layer_type == 'Conv2D':
                layers.append({'type': 'Conv2D', 'filters': params[2], 'kernel': params[3]})
            elif layer_type == 'MaxPool2D':
                layers.append({'type': 'MaxPool2D', 'pool_size': params[4]})
            elif layer_type == 'Dense':
                layers.append({'type': 'Dense', 'units': params[5]})
            elif layer_type == 'Dropout':
                layers.append({'type': 'Dropout', 'rate': params[6]})
            elif layer_type == 'BatchNorm2D':
                layers.append({'type': 'BatchNorm2D'})
            elif layer_type == 'LayerNorm':
                layers.append({'type': 'LayerNorm'})
            elif layer_type == 'Residual':
                if len(layers) > 1:
                    layers.append({'type': 'Residual', 'from': max(0, len(layers)-2)})
            elif layer_type == 'Skip':
                if len(layers) > 1:
                    layers.append({'type': 'Skip', 'from': max(0, len(layers)-2)})
        layers.append({'type': 'Dense', 'units': 10})
        hparams = {
            'learning_rate': params[7],
            'optimizer': params[8],
            'batch_size': params[9]
        }
        arch = {'layers': layers, 'hparams': hparams}
        self.arch_queue.append(params)
        self.asked += 1
        return arch

    def update_with_result(self, arch, metrics):
        # Use validation accuracy minus parameter penalty as the objective
        score = metrics['val_acc'] - 1e-6 * metrics['num_params']
        if self.arch_queue:
            params = self.arch_queue.pop(0)
            self.optimizer.tell(params, -score)  # skopt minimizes, so use -score
        self.results.append({'arch': arch, 'metrics': metrics, 'score': score})

# --- Reinforcement Search (Stub) ---
class ReinforcementSearchStrategy(SearchStrategy):
    def generate_next_architecture(self):
        raise NotImplementedError('Reinforcement search not implemented yet.')

# --- Grid/Manual Search (Stub) ---
class GridSearchStrategy(SearchStrategy):
    def generate_next_architecture(self):
        raise NotImplementedError('Grid/manual search not implemented yet.')

# --- Optuna Search ---
class OptunaSearchStrategy(SearchStrategy):
    def __init__(self, search_space, n_trials=20, **kwargs):
        super().__init__(search_space)
        self.n_trials = n_trials
        self.study = optuna.create_study(direction='maximize')
        self.trial_queue = []
        self.results = []
        self._prepare_trials()
        self.current_trial = None
    def _prepare_trials(self):
        def objective(trial):
            # Sample architecture
            num_layers = trial.suggest_int('num_layers', 2, 5)
            layers = []
            for i in range(num_layers):
                layer_type = trial.suggest_categorical(f'layer_type_{i}', self.search_space['LAYER_TYPES'])
                if layer_type == 'Conv2D':
                    layers.append({
                        'type': 'Conv2D',
                        'filters': trial.suggest_categorical(f'filters_{i}', self.search_space['CONV_FILTERS']),
                        'kernel': trial.suggest_categorical(f'kernel_{i}', self.search_space['KERNEL_SIZES'])
                    })
                elif layer_type == 'MaxPool2D':
                    layers.append({
                        'type': 'MaxPool2D',
                        'pool_size': trial.suggest_categorical(f'pool_{i}', self.search_space['POOL_SIZES'])
                    })
                elif layer_type == 'Dense':
                    layers.append({
                        'type': 'Dense',
                        'units': trial.suggest_categorical(f'units_{i}', self.search_space['DENSE_UNITS'])
                    })
                elif layer_type == 'Dropout':
                    layers.append({
                        'type': 'Dropout',
                        'rate': trial.suggest_categorical(f'dropout_{i}', self.search_space['DROPOUT_RATES'])
                    })
            layers.append({'type': 'Dense', 'units': 10})
            # Sample hparams
            hparams = {
                'learning_rate': trial.suggest_categorical('learning_rate', self.search_space['LEARNING_RATES']),
                'optimizer': trial.suggest_categorical('optimizer', self.search_space['OPTIMIZERS']),
                'batch_size': trial.suggest_categorical('batch_size', self.search_space['BATCH_SIZES'])
            }
            arch = {'layers': layers, 'hparams': hparams}
            self.trial_queue.append((trial, arch))
            return 0.0  # Dummy, real eval happens outside
        # Pre-populate trial queue
        for _ in range(self.n_trials):
            self.study.optimize(objective, n_trials=1, catch=(Exception,))
    def generate_next_architecture(self):
        if not self.trial_queue:
            return None
        self.current_trial, arch = self.trial_queue.pop(0)
        return arch
    def update_with_result(self, arch, metrics):
        # Report result to Optuna
        if self.current_trial is not None:
            score = metrics['val_acc'] - 1e-6 * metrics['num_params']
            self.current_trial.report(score, step=0)
            self.results.append({'arch': arch, 'metrics': metrics, 'score': score})
            self.current_trial = None

LATENCY_LOOKUP = {
    'edge_gpu': {
        'Conv2D': lambda in_c, out_c, k, h, w: 2 * in_c * out_c * k * k * h * w / 1e9,  # GFLOPs, dummy
        'Dense': lambda in_f, out_f: 2 * in_f * out_f / 1e9,  # GFLOPs, dummy
        'MaxPool2D': lambda h, w, p: 0.00001 * h * w,  # dummy
        'Dropout': lambda h, w: 0.0
    },
    'raspberry_pi': {
        'Conv2D': lambda in_c, out_c, k, h, w: 4 * in_c * out_c * k * k * h * w / 1e9,  # slower
        'Dense': lambda in_f, out_f: 4 * in_f * out_f / 1e9,
        'MaxPool2D': lambda h, w, p: 0.00002 * h * w,
        'Dropout': lambda h, w: 0.0
    }
}

class LatencyEstimator:
    def __init__(self, device='edge_gpu', input_shape=(1, 28, 28)):
        self.device = device
        self.input_shape = input_shape
    def estimate(self, arch):
        lookup = LATENCY_LOOKUP[self.device]
        h, w = self.input_shape[1:]
        in_c = 1
        total = 0.0
        for layer in arch['layers']:
            if layer['type'] == 'Conv2D':
                out_c = layer['filters']
                k = layer['kernel']
                total += lookup['Conv2D'](in_c, out_c, k, h, w)
                in_c = out_c
            elif layer['type'] == 'MaxPool2D':
                p = layer['pool_size']
                h //= p
                w //= p
                total += lookup['MaxPool2D'](h, w, p)
            elif layer['type'] == 'Dense':
                in_f = h * w * in_c
                out_f = layer['units']
                total += lookup['Dense'](in_f, out_f)
                in_c = 1
                h = w = 1
            elif layer['type'] == 'Dropout':
                total += lookup['Dropout'](h, w)
        return total

# Update NSGA2SearchStrategy to support latency as an objective
class NSGA2SearchStrategy(SearchStrategy):
    def __init__(self, search_space, pop_size=10, generations=10, target_device=None, multiobj=('val_acc', 'num_params'), **kwargs):
        super().__init__(search_space)
        self.pop_size = pop_size
        self.generations = generations
        self.target_device = target_device or 'edge_gpu'
        self.multiobj = multiobj
        self.latency_estimator = LatencyEstimator(self.target_device)
        self.population = [random_architecture(self.search_space) for _ in range(self.pop_size)]
        self.metrics = []
        self.current_gen = 0
        self.eval_queue = self.population.copy()
        self.fronts = []
    def generate_next_architecture(self):
        if self.eval_queue:
            return self.eval_queue.pop(0)
        return None
    def update_with_result(self, arch, metrics):
        # Add latency if needed
        if 'latency' not in metrics and 'latency' in self.multiobj:
            metrics['latency'] = self.latency_estimator.estimate(arch)
        self.metrics.append({'arch': arch, 'metrics': metrics})
        if len(self.metrics) % self.pop_size == 0:
            # End of generation: perform NSGA-II selection
            objs = []
            for m in self.metrics[-self.pop_size:]:
                obj = []
                for o in self.multiobj:
                    v = m['metrics'][o] if o != 'latency' else m['metrics'].get('latency', 0.0)
                    if o == 'val_acc':
                        obj.append(v)  # maximize
                    else:
                        obj.append(-v)  # minimize
                objs.append(obj)
            objs = np.array(objs)
            fronts = nsga2_fast_non_dominated_sort(objs)
            pareto_front = [self.metrics[-self.pop_size:][i]['arch'] for i in fronts[0]]
            self.fronts.append(pareto_front)
            # Evolve next generation
            next_pop = []
            for idx in fronts[0]:
                next_pop.append(self.metrics[-self.pop_size:][idx]['arch'])
            while len(next_pop) < self.pop_size:
                parent = np.random.choice(next_pop)
                child = mutate_arch(parent, self.search_space)
                next_pop.append(child)
            self.population = next_pop
            self.eval_queue = self.population.copy()
            self.current_gen += 1
    def get_pareto_front(self):
        if self.fronts:
            return self.fronts[-1]
        return []

class OneShotNASSupernet:
    def __init__(self, search_space, input_shape=(1, 28, 28)):
        self.search_space = search_space
        self.input_shape = input_shape
        # For demo: just store all possible layers/ops
        self.layers = []
        for layer_type in search_space['LAYER_TYPES']:
            if layer_type == 'Conv2D':
                for f in search_space['CONV_FILTERS']:
                    for k in search_space['KERNEL_SIZES']:
                        self.layers.append({'type': 'Conv2D', 'filters': f, 'kernel': k})
            elif layer_type == 'MaxPool2D':
                for p in search_space['POOL_SIZES']:
                    self.layers.append({'type': 'MaxPool2D', 'pool_size': p})
            elif layer_type == 'Dense':
                for u in search_space['DENSE_UNITS']:
                    self.layers.append({'type': 'Dense', 'units': u})
            elif layer_type == 'Dropout':
                for r in search_space['DROPOUT_RATES']:
                    self.layers.append({'type': 'Dropout', 'rate': r})
        self.layers.append({'type': 'Dense', 'units': 10})
        # In a real supernet, this would be a PyTorch model with all ops
        self.weights = {}  # Placeholder for shared weights
    def sample_subnet(self):
        # Randomly sample a valid sub-network (subset of layers)
        num_layers = np.random.randint(2, 6)
        idxs = np.random.choice(len(self.layers)-1, num_layers, replace=False)
        subnet = [self.layers[i] for i in sorted(idxs)] + [self.layers[-1]]
        return subnet
    def evaluate_subnet(self, subnet):
        # For demo: return a random accuracy using shared weights
        # In real code, would forward pass using only subnet layers and shared weights
        return {'val_acc': np.random.uniform(0.8, 0.99), 'num_params': np.random.randint(10000, 100000)}

class OneShotSearchStrategy(SearchStrategy):
    def __init__(self, search_space, n_samples=20, **kwargs):
        super().__init__(search_space)
        self.supernet = OneShotNASSupernet(search_space)
        self.n_samples = n_samples
        self.samples = 0
        self.results = []
    def generate_next_architecture(self):
        if self.samples < self.n_samples:
            subnet = self.supernet.sample_subnet()
            self.current_subnet = subnet
            return {'layers': subnet, 'hparams': {}}  # hparams can be added
        return None
    def update_with_result(self, arch, metrics):
        self.results.append({'arch': arch, 'metrics': metrics})
        self.samples += 1
    def evaluate_current(self):
        # Evaluate current subnet using shared weights
        return self.supernet.evaluate_subnet(self.current_subnet)

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
    elif name == 'optuna':
        return OptunaSearchStrategy(search_space, **kwargs)
    elif name == 'nsga2':
        return NSGA2SearchStrategy(search_space, **kwargs)
    elif name == 'oneshot':
        return OneShotSearchStrategy(search_space, **kwargs)
    else:
        raise ValueError(f'Unknown strategy: {name}')

def get_search_space(template=None, required_modules=None):
    space = {
        'LAYER_TYPES': ['Conv2D', 'MaxPool2D', 'Dense', 'Dropout', 'Residual', 'Skip', 'BatchNorm2D', 'LayerNorm'],
        'CONV_FILTERS': [16, 32, 64],
        'KERNEL_SIZES': [3, 5],
        'POOL_SIZES': [2],
        'DENSE_UNITS': [64, 128, 256],
        'DROPOUT_RATES': [0.2, 0.3, 0.5],
        'LEARNING_RATES': [0.001, 0.0005, 0.0001],
        'OPTIMIZERS': ['adam', 'sgd'],
        'BATCH_SIZES': [32, 64, 128],
        'TEMPLATE': template,
        'REQUIRED_MODULES': required_modules
    }
    return space

# --- Utility: random_architecture, mutate_arch, score_arch ---
def random_architecture(search_space):
    LAYER_TYPES = search_space['LAYER_TYPES']
    CONV_FILTERS = search_space['CONV_FILTERS']
    KERNEL_SIZES = search_space['KERNEL_SIZES']
    POOL_SIZES = search_space['POOL_SIZES']
    DENSE_UNITS = search_space['DENSE_UNITS']
    DROPOUT_RATES = search_space['DROPOUT_RATES']
    LEARNING_RATES = search_space.get('LEARNING_RATES', [0.001])
    OPTIMIZERS = search_space.get('OPTIMIZERS', ['adam'])
    BATCH_SIZES = search_space.get('BATCH_SIZES', [64])
    TEMPLATE = search_space.get('TEMPLATE')
    REQUIRED_MODULES = search_space.get('REQUIRED_MODULES')
    layers = []
    num_layers = random.randint(2, 5)
    for i in range(num_layers):
        layer_type = random.choice(LAYER_TYPES)
        if layer_type == 'Conv2D':
            layers.append({'type': 'Conv2D', 'filters': random.choice(CONV_FILTERS), 'kernel': random.choice(KERNEL_SIZES)})
            if REQUIRED_MODULES and 'BatchNorm2D' in REQUIRED_MODULES:
                layers.append({'type': 'BatchNorm2D'})
            if REQUIRED_MODULES and 'LayerNorm' in REQUIRED_MODULES:
                layers.append({'type': 'LayerNorm'})
        elif layer_type == 'MaxPool2D':
            layers.append({'type': 'MaxPool2D', 'pool_size': random.choice(POOL_SIZES)})
        elif layer_type == 'Dense':
            layers.append({'type': 'Dense', 'units': random.choice(DENSE_UNITS)})
        elif layer_type == 'Dropout':
            layers.append({'type': 'Dropout', 'rate': random.choice(DROPOUT_RATES)})
        elif layer_type == 'Residual' and len(layers) > 1:
            # Residual connection from previous layer
            layers.append({'type': 'Residual', 'from': random.randint(0, len(layers)-1)})
        elif layer_type == 'Skip' and len(layers) > 1:
            # Skip connection from any previous layer
            layers.append({'type': 'Skip', 'from': random.randint(0, len(layers)-1)})
        elif layer_type == 'BatchNorm2D':
            layers.append({'type': 'BatchNorm2D'})
        elif layer_type == 'LayerNorm':
            layers.append({'type': 'LayerNorm'})
    layers.append({'type': 'Dense', 'units': 10})
    # Insert template if provided
    if TEMPLATE:
        for t_layer in TEMPLATE:
            if t_layer not in layers:
                layers.insert(random.randint(0, len(layers)-1), t_layer)
    # Ensure required modules are present
    if REQUIRED_MODULES:
        for req in REQUIRED_MODULES:
            if not any(l['type'] == req for l in layers):
                layers.insert(random.randint(0, len(layers)-1), {'type': req})
    # Sample hyperparameters
    hparams = {
        'learning_rate': random.choice(LEARNING_RATES),
        'optimizer': random.choice(OPTIMIZERS),
        'batch_size': random.choice(BATCH_SIZES)
    }
    return {'layers': layers, 'hparams': hparams}

def mutate_arch(arch, search_space):
    new_arch = json.loads(json.dumps(arch))
    layers = new_arch['layers']
    LAYER_TYPES = search_space['LAYER_TYPES']
    CONV_FILTERS = search_space['CONV_FILTERS']
    KERNEL_SIZES = search_space['KERNEL_SIZES']
    POOL_SIZES = search_space['POOL_SIZES']
    DENSE_UNITS = search_space['DENSE_UNITS']
    DROPOUT_RATES = search_space['DROPOUT_RATES']
    LEARNING_RATES = search_space.get('LEARNING_RATES', [0.001])
    OPTIMIZERS = search_space.get('OPTIMIZERS', ['adam'])
    BATCH_SIZES = search_space.get('BATCH_SIZES', [64])
    TEMPLATE = search_space.get('TEMPLATE')
    REQUIRED_MODULES = search_space.get('REQUIRED_MODULES')
    op = random.choice(['add', 'remove', 'change', 'hparam'])
    if op == 'add' and len(layers) < 8:
        idx = random.randint(0, len(layers)-2)
        layer_type = random.choice(LAYER_TYPES)
        if layer_type == 'Conv2D':
            new_layer = {'type': 'Conv2D', 'filters': random.choice(CONV_FILTERS), 'kernel': random.choice(KERNEL_SIZES)}
            if REQUIRED_MODULES and 'BatchNorm2D' in REQUIRED_MODULES:
                layers.insert(idx+1, {'type': 'BatchNorm2D'})
            if REQUIRED_MODULES and 'LayerNorm' in REQUIRED_MODULES:
                layers.insert(idx+1, {'type': 'LayerNorm'})
        elif layer_type == 'MaxPool2D':
            new_layer = {'type': 'MaxPool2D', 'pool_size': random.choice(POOL_SIZES)}
        elif layer_type == 'Dense':
            new_layer = {'type': 'Dense', 'units': random.choice(DENSE_UNITS)}
        elif layer_type == 'Dropout':
            new_layer = {'type': 'Dropout', 'rate': random.choice(DROPOUT_RATES)}
        elif layer_type == 'Residual' and len(layers) > 1:
            new_layer = {'type': 'Residual', 'from': random.randint(0, len(layers)-1)}
        elif layer_type == 'Skip' and len(layers) > 1:
            new_layer = {'type': 'Skip', 'from': random.randint(0, len(layers)-1)}
        elif layer_type == 'BatchNorm2D':
            new_layer = {'type': 'BatchNorm2D'}
        elif layer_type == 'LayerNorm':
            new_layer = {'type': 'LayerNorm'}
        else:
            new_layer = None
        if new_layer:
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
        elif layer['type'] == 'Residual' and len(layers) > 1:
            layer['from'] = random.randint(0, idx-1) if idx > 0 else 0
        elif layer['type'] == 'Skip' and len(layers) > 1:
            layer['from'] = random.randint(0, idx-1) if idx > 0 else 0
    elif op == 'hparam':
        # Mutate a hyperparameter
        hp = random.choice(['learning_rate', 'optimizer', 'batch_size'])
        if hp == 'learning_rate':
            new_arch['hparams']['learning_rate'] = random.choice(LEARNING_RATES)
        elif hp == 'optimizer':
            new_arch['hparams']['optimizer'] = random.choice(OPTIMIZERS)
        elif hp == 'batch_size':
            new_arch['hparams']['batch_size'] = random.choice(BATCH_SIZES)
    # Ensure template and required modules are present
    if TEMPLATE:
        for t_layer in TEMPLATE:
            if t_layer not in layers:
                layers.insert(random.randint(0, len(layers)-1), t_layer)
    if REQUIRED_MODULES:
        for req in REQUIRED_MODULES:
            if not any(l['type'] == req for l in layers):
                layers.insert(random.randint(0, len(layers)-1), {'type': req})
    return new_arch

def score_arch(metrics, constraint_parser=None, arch=None):
    # Higher accuracy, smaller model is better
    score = metrics['val_acc'] - 1e-6 * metrics['num_params']
    if constraint_parser and arch is not None:
        penalty = constraint_parser.penalty(arch, metrics)
        score -= penalty
    return score

class ConstraintParser:
    def __init__(self, constraints=None):
        self.constraints = constraints or {}
    def check(self, arch, metrics):
        # Returns True if all constraints are satisfied
        if 'max_params' in self.constraints:
            if metrics.get('num_params', 0) > self.constraints['max_params']:
                return False
        if 'max_flops' in self.constraints:
            if metrics.get('flops', 0) > self.constraints['max_flops']:
                return False
        if 'min_accuracy' in self.constraints:
            if metrics.get('val_acc', 0) < self.constraints['min_accuracy']:
                return False
        return True
    def penalty(self, arch, metrics):
        # Returns a penalty score (0 if satisfied, >0 if violated)
        penalty = 0
        if 'max_params' in self.constraints:
            excess = metrics.get('num_params', 0) - self.constraints['max_params']
            if excess > 0:
                penalty += excess / self.constraints['max_params']
        if 'max_flops' in self.constraints:
            excess = metrics.get('flops', 0) - self.constraints['max_flops']
            if excess > 0:
                penalty += excess / self.constraints['max_flops']
        if 'min_accuracy' in self.constraints:
            deficit = self.constraints['min_accuracy'] - metrics.get('val_acc', 0)
            if deficit > 0:
                penalty += deficit / self.constraints['min_accuracy']
        return penalty 

def nsga2_fast_non_dominated_sort(objs):
    # objs: N x 2 array (maximize first, minimize second)
    N = objs.shape[0]
    S = [[] for _ in range(N)]
    n = [0] * N
    rank = [0] * N
    fronts = [[]]
    for p in range(N):
        S[p] = []
        n[p] = 0
        for q in range(N):
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

def dominates(a, b):
    # a, b: 2D objectives (maximize a[0], minimize a[1])
    return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1]) 

class NASBench101Backend:
    def __init__(self):
        # In real code, load NASBench-101 dataset
        self.db = {}  # Placeholder
    def evaluate(self, arch):
        # Simulate evaluation: hash arch to get a deterministic score
        h = hash(str(arch))
        acc = 0.8 + (h % 100) / 1000.0  # 0.8-0.9
        params = 10000 + (h % 1000)
        return {'val_acc': acc, 'num_params': params}

class NASBench201Backend:
    def __init__(self):
        # In real code, load NASBench-201 dataset
        self.db = {}  # Placeholder
    def evaluate(self, arch):
        h = hash(str(arch))
        acc = 0.85 + (h % 100) / 1000.0  # 0.85-0.95
        params = 15000 + (h % 2000)
        return {'val_acc': acc, 'num_params': params} 