from haven import haven_utils as hu

EXP_GROUPS = {}
EXP_GROUPS['mnist_classification'] = { 
        "batch_size": 256,
        "learning_rate": [5e-4],
        "dataset": 'mnist',
        "model": 'classifier',
        "backbone": "conv4",
        "conv4": {
            "in_ch": 1,
        },
        "max_epochs": 50,
        "seed": [1,2,3],
        "head": "linear",
        "output_dim": 10
    }
EXP_GROUPS['double_mnist_classification'] = { 
        "batch_size": 256,
        "learning_rate": [5e-4],
        "dataset": 'double_mnist',
        "model": 'double_classifier',
        "backbone": "conv4",
        "max_epochs": 50,
        "seed": [1,2,3],
        "head": "linear",
        "conv4_in_ch": 3,
        "output_dim": 20,
        "conv4": {"in_ch": 3}
    }
EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}