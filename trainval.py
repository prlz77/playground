from datasets import get_dataset
from models import get_model
import pytorch_lightning as pl
import argparse
import exp_configs
from utils import Bunch
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
import datetime
from utils import load_json
from hashlib import md5
import os
import shutil as sh

def trainval(exp_dict, savedir, args):
    pl.seed_everything(exp_dict['seed'], workers=True)
    hparams = Bunch(exp_dict)
    dm = get_dataset(hparams, args)
    model = get_model(hparams)
    last_checkpoint = None
    checkpoints = glob.glob(f'{savedir}/checkpoints/last*.ckpt')
    if len(checkpoints) > 0:
        last_checkpoint = list(sorted(checkpoints))[-1]
        with (Path(savedir) / "timestamp.txt").open('r') as infile:
            timestamp = infile.read()
        resume = True
    else:
        timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(":","-")
        with (Path(savedir) / "timestamp.txt").open('w') as infile:
            infile.write(timestamp)
        resume = False
    log_name = f"{Path(savedir).name}_{timestamp}"

    checkpoint_best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath=str(Path(savedir) / "checkpoints"),
        filename="last-{epoch:02d}-{val_loss:.2f}",
    )
    checkpoint_last_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=str(Path(savedir) / "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.2f}",
    )
    logger = WandbLogger(project=Path(savedir).parent.name, 
        name=log_name, 
        id=log_name, 
        group=args.exp_group, 
        save_dir=savedir, 
        resume=resume)
    # logger = TensorBoardLogger(savedir, log_name)
    trainer = pl.Trainer(max_epochs=hparams["max_epochs"], accelerator="gpu", devices=1, default_root_dir=str(Path(savedir).parent.parent), logger=logger, resume_from_checkpoint=last_checkpoint,
                            callbacks=[checkpoint_last_callback, checkpoint_best_callback], precision=16)
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True,
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument("-r", "--reset", default=0, type=int, help="Reset or resume the experiment.")
    parser.add_argument("--num_workers", type=int, default=1)

    args, others = parser.parse_known_args()

    
    exp_configs = exp_configs.EXP_GROUPS[args.exp_group]
    exp_id = md5(str(exp_configs[0]).encode()).hexdigest()
    exp_path = str(Path(args.savedir_base) / exp_id)
    if args.reset == 1:
        sh.rmtree(exp_path, ignore_errors=True)
    
    os.makedirs(exp_path, exist_ok=True)
    trainval(exp_configs[0], savedir=exp_path, args=args)
