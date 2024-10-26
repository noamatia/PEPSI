import os
import wandb
import torch
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from pepsi.consts import *
from pepsi.model import PEPSI
from shapetalk import ShapeTalk
from pepsi.custom_types import *
from torch.utils.data import DataLoader
from point_e.util.point_cloud import PointNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--wandb_api_key", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="PEPSI")
    return parser.parse_args()


def main(args: argparse.Namespace, device: torch.device):
    os.environ[WANDB_API_KEY] = args.wandb_api_key
    run = wandb.Api().run(os.path.join(args.wandb_project, args.run_id))
    config = run.config
    shape_category = SHAPE_CATEGORY[config[SHAPE_CATEGORY_CONF_KEY]]
    utterance_key = UTTERANCE_KEY[config[UTTERANCE_KEY_CONF_KEY]]

    base_dir = config[BASE_DIR]
    output_dir = os.path.join(base_dir, RUNS, run.name)
    checkpoints_dir = os.path.join(output_dir, CHECKPOINTS)
    pointnet_models_dir = os.path.join(base_dir, POINTNET_MODELS)
    pcs_dir = os.path.join(base_dir, PCS)
    images_dir = os.path.join(base_dir, IMAGES)
    os.makedirs(images_dir, exist_ok=True)
    if args.checkpoint:
        checkpoint = args.checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
    else:
        checkpoint_path = sorted(
            [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir)],
            key=os.path.getmtime,
            reverse=True,
        )[0]
        checkpoint = os.path.basename(checkpoint_path)
    checkpoint = checkpoint.removesuffix(CKPT).replace("=", "_").replace("-", "_")
    results_pcs_dir = os.path.join(output_dir, PCS, checkpoint)
    os.makedirs(results_pcs_dir, exist_ok=True)
    results_images_dir = os.path.join(output_dir, IMAGES, checkpoint)
    os.makedirs(results_images_dir, exist_ok=True)
    injections_dir = os.path.join(output_dir, INJECTIONS, checkpoint)

    shapetalk_df = pd.read_csv(SHAPETALK_CSV_PATH, index_col=ID)
    shapetalk_df = shapetalk_df[
        shapetalk_df.source_object_class == shape_category.value
    ]
    if config[LOCAL]:
        shapetalk_df = shapetalk_df[shapetalk_df.is_local]
    test_df = shapetalk_df[shapetalk_df.changeit_split == SPLIT.TEST.value]

    pointnet = PointNet(
        shape_category=shape_category,
        pointnet_models_dir=pointnet_models_dir,
    )

    test_dataset = ShapeTalk(
        df=test_df,
        device=device,
        pcs_dir=pcs_dir,
        split=SPLIT.TEST,
        pointnet=pointnet,
        images_dir=images_dir,
        uid_key=UID_KEY.SOURCE_UID,
        utterance_key=utterance_key,
        batch_size=config[BATCH_SIZE],
        shapenet_dir=config[SHAPENET_DIR],
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config[BATCH_SIZE],
    )

    model = PEPSI.load_from_checkpoint(
        dev=device,
        lr=config[LR],
        log_wandb=False,
        copy_prob=config[COPY_PROB],
        batch_size=config[BATCH_SIZE],
        copy_prompt=config[COPY_PROMPT],
        checkpoint_path=checkpoint_path,
        cond_drop_prob=config[COND_DROP_PROB],
    )
    model.eval()

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        indices, prompts, source_latents, target_latents = (
            batch[INDICES],
            batch[PROMPTS],
            batch[SOURCE_LATENTS],
            batch[TARGET_LATENTS],
        )

        samples = model.sampler.sample_batch(
            batch_size=config[BATCH_SIZE],
            model_kwargs={TEXTS: prompts},
            guidances=[source_latents, None],
        )
        pcs = model.sampler.output_to_point_clouds(samples)
        for idx, pc in zip(indices, pcs):
            pc.save(os.path.join(results_pcs_dir, f"{idx}_{N_PTS_SAMPLE}.npz"))
            pc.render(os.path.join(results_images_dir, f"{idx}.png"))

        if os.path.exists(injections_dir):
            shutil.rmtree(injections_dir)
        os.makedirs(injections_dir, exist_ok=True)

        samples = model.sampler.sample_batch(
            injections_dir=injections_dir,
            batch_size=config[BATCH_SIZE],
            guidances=[target_latents, None],
            model_kwargs={TEXTS: [config[COPY_PROMPT]] * config[BATCH_SIZE]},
        )
        pcs = model.sampler.output_to_point_clouds(samples)
        for idx, pc in zip(indices, pcs):
            pc.save(os.path.join(results_pcs_dir, f"{idx}_copy_{N_PTS_SAMPLE}.npz"))
            pc.render(os.path.join(results_images_dir, f"{idx}_copy.png"))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    main(args, device)
