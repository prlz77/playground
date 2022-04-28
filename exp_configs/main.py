from haven import haven_utils as hu

EXP_GROUPS = {}
EXP_GROUPS['mnist_classification'] = { 
        "batch_size": 256,
        "learning_rate": [5e-4],
        "dataset": 'mnist',
        "model": 'classifier',
        "backbone": "conv4",
        "max_epochs": 50,
        "seed": [1,2,3],
        "head": "linear",
        "output_dim": 10
    }
EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}