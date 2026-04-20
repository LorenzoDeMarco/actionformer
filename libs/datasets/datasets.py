import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed

datasets = {}

def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def make_dataset(name, is_training, split, **kwargs):
    kwargs.pop('backbone', None)
    kwargs.pop('division_type', None)
    kwargs.pop('videos_type', None)
    
    dataset = datasets[name](is_training, split, **kwargs)
    return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader