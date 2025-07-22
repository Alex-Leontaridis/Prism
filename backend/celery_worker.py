import os
from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models import Base, NASJob, NASResult
import json
from backend.nas_strategies import get_search_space, get_search_strategy

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///nas.db')

celery = Celery('celery_worker', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

@celery.task(bind=True)
def run_nas_job(self, job_id):
    session = SessionLocal()
    job = session.query(NASJob).filter_by(job_id=job_id).first()
    if not job:
        return {'error': 'Job not found'}
    # Set random seed
    import random, torch, numpy as np
    random.seed(job.seed)
    torch.manual_seed(job.seed)
    np.random.seed(job.seed)
    # Load config
    config = job.config
    search_space = get_search_space()
    strategy_name = config.get('strategy', 'random')
    if strategy_name == 'optuna':
        from backend.nas_strategies import OptunaSearchStrategy
        n_trials = config.get('n_trials', 20)
        strategy = OptunaSearchStrategy(search_space, n_trials=n_trials)
        step = 0
        while True:
            arch = strategy.generate_next_architecture()
            if arch is None:
                break
            # Dummy train_and_eval for demo
            metrics = {'val_acc': 0.8 + 0.01*step, 'num_params': 10000 + 100*step}
            strategy.update_with_result(arch, metrics)
            result = NASResult(
                job_id=job_id,
                step=step,
                architecture=arch['layers'],
                hparams=arch['hparams'],
                metrics=metrics
            )
            session.add(result)
            step += 1
        job.status = 'completed'
        session.commit()
        session.close()
        return {'status': 'done', 'job_id': job_id, 'strategy': 'optuna'}
    else:
        # Existing logic for other strategies
        for step in range(3):
            result = NASResult(
                job_id=job_id,
                step=step,
                architecture={'layers': []},
                hparams={'lr': 0.001},
                metrics={'val_acc': 0.9 + 0.01*step}
            )
            session.add(result)
        job.status = 'completed'
        session.commit()
        session.close()
        return {'status': 'done', 'job_id': job_id} 