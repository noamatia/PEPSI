import os
import wandb
import torch
import argparse
import pandas as pd
from pepsi.consts import *
from pepsi.model import PEPSI
from datetime import datetime
import pytorch_lightning as pl
from shapetalk import ShapeTalk
from pepsi.custom_types import *
from point_e.util.point_cloud import PointNet
from pepsi.shapetalk import utterance_key_type
from torch.utils.data import DataLoader, ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--shapenet_dir", type=str)
    parser.add_argument("--wandb_api_key", type=str)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--val_freq", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--copy_prob", type=float, default=0.1)
    parser.add_argument("--copy_prompt", type=str, default="COPY")
    parser.add_argument("--num_val_samples", type=int, default=10)
    parser.add_argument("--utterance_key", type=utterance_key_type)
    parser.add_argument("--cond_drop_prob", type=float, default=0.5)
    parser.add_argument("--wandb_project", type=str, default="PEPSI")
    parser.add_argument("--shape_category", type=shape_category_type)
    return parser.parse_args()


def build_name(args: argparse.Namespace) -> str:
    name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    name += f"_{args.shape_category.value}"
    name += f"_{args.utterance_key.value}"
    if args.local:
        name += "_local"
    name += f"_{args.copy_prompt}_{args.copy_prob}"
    return name


def main(args: argparse.Namespace, device: torch.device):
    log_wandb = args.wandb_api_key is not None and args.wandb_project is not None
    if log_wandb:
        os.environ[WANDB_API_KEY] = args.wandb_api_key

    if args.run_id is not None:
        assert log_wandb, "Must provide WANDB_API_KEY to resume run"
        run_id, wandb_project = args.run_id, args.wandb_project
        run = wandb.Api().run(os.path.join(wandb_project, run_id))
        name, config = run.name, run.config
        assert wandb_project == config[WANDB_PROJECT], "Project mismatch"
        config[SHAPE_CATEGORY_CONF_KEY] = SHAPE_CATEGORY[
            config[SHAPE_CATEGORY_CONF_KEY]
        ]
        config[UTTERANCE_KEY_CONF_KEY] = UTTERANCE_KEY[config[UTTERANCE_KEY_CONF_KEY]]
        args = argparse.Namespace(**run.config)
        output_dir = os.path.join(args.base_dir, RUNS, name)
        checkpoints_dir = os.path.join(output_dir, CHECKPOINTS)
        checkpoint_path = sorted(
            [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir)],
            key=os.path.getmtime,
            reverse=True,
        )[0]
        os.environ[WANDB_DIR] = output_dir
        wandb.init(project=wandb_project, id=run_id, resume="allow")
    else:
        name = build_name(args)
        output_dir = os.path.join(args.base_dir, RUNS, name)
        os.makedirs(output_dir, exist_ok=True)
        checkpoints_dir = os.path.join(output_dir, CHECKPOINTS)
        checkpoint_path = None
        if log_wandb:
            os.environ[WANDB_DIR] = output_dir
            wandb.init(project=args.wandb_project, name=name, config=vars(args))

    shapetalk_df = pd.read_csv(SHAPETALK_CSV_PATH, index_col=ID)
    shapetalk_df = shapetalk_df[
        shapetalk_df.source_object_class == args.shape_category.value
    ]
    if args.local:
        shapetalk_df = shapetalk_df[shapetalk_df.is_local]
    train_df = shapetalk_df[shapetalk_df.changeit_split == SPLIT.TRAIN.value]
    test_df = shapetalk_df[shapetalk_df.changeit_split == SPLIT.TEST.value]

    pcs_dir = os.path.join(args.base_dir, PCS)
    os.makedirs(pcs_dir, exist_ok=True)
    pointnet_models_dir = os.path.join(args.base_dir, POINTNET_MODELS)
    os.makedirs(pointnet_models_dir, exist_ok=True)

    pointnet = PointNet(
        shape_category=args.shape_category,
        pointnet_models_dir=pointnet_models_dir,
    )

    train_dataset = ShapeTalk(
        df=train_df,
        device=device,
        pcs_dir=pcs_dir,
        split=SPLIT.TRAIN,
        pointnet=pointnet,
        uid_key=UID_KEY.TARGET_UID,
        batch_size=args.batch_size,
        shapenet_dir=args.shapenet_dir,
        utterance_key=args.utterance_key,
    )
    train_dataloader = DataLoader(
        shuffle=True,
        dataset=train_dataset,
        batch_size=args.batch_size,
    )

    val_dataset_train = ShapeTalk(
        device=device,
        pcs_dir=pcs_dir,
        split=SPLIT.TRAIN,
        pointnet=pointnet,
        uid_key=UID_KEY.TARGET_UID,
        shapenet_dir=args.shapenet_dir,
        batch_size=args.num_val_samples,
        utterance_key=args.utterance_key,
        df=train_df.head(args.num_val_samples),
    )
    val_dataset_test = ShapeTalk(
        device=device,
        pcs_dir=pcs_dir,
        split=SPLIT.TEST,
        pointnet=pointnet,
        uid_key=UID_KEY.TARGET_UID,
        shapenet_dir=args.shapenet_dir,
        batch_size=args.num_val_samples,
        utterance_key=args.utterance_key,
        df=test_df.head(args.num_val_samples),
    )
    val_dataloader = DataLoader(
        batch_size=args.num_val_samples * 2,
        dataset=ConcatDataset([val_dataset_train, val_dataset_test]),
    )

    model_kwargs = {
        LR: args.lr,
        DEV: device,
        LOG_WANDB: log_wandb,
        COPY_PROB: args.copy_prob,
        BATCH_SIZE: args.batch_size,
        COPY_PROMPT: args.copy_prompt,
        VAL_DATALOADER: val_dataloader,
        COND_DROP_PROB: args.cond_drop_prob,
    }
    if checkpoint_path is not None:
        model_kwargs[VAL_DATALOADER] = None
        model_kwargs[CHECKPOINT_PATH] = checkpoint_path
        model = PEPSI.load_from_checkpoint(**model_kwargs)
    else:
        model = PEPSI(**model_kwargs)

    if log_wandb:
        wandb.watch(model)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        dirpath=checkpoints_dir,
        every_n_epochs=args.val_freq,
    )
    logger = [TensorBoardLogger(output_dir)]
    if log_wandb:
        logger.append(WandbLogger())
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accumulate_grad_batches=10,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.val_freq,
    )
    trainer.fit(
        model=model,
        ckpt_path=checkpoint_path,
        val_dataloaders=val_dataloader,
        train_dataloaders=train_dataloader,
    )

    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    main(args, device)
