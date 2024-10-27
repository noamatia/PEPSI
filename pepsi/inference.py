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
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--copy_steps_list", type=str, default="0,35,45,55")
    return parser.parse_args()


def main(args: argparse.Namespace, device: torch.device):
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
    injection_dir = os.path.join(output_dir, INJECTIONS, checkpoint)

    copy_steps_list = list(map(int, args.copy_steps_list.split(",")))

    shapetalk_df = pd.read_csv(SHAPETALK_CSV_PATH, index_col=ID)
    shapetalk_df = shapetalk_df[
        shapetalk_df.source_object_class == shape_category.value
    ]
    if config[LOCAL]:
        shapetalk_df = shapetalk_df[shapetalk_df.is_local]
    test_df = shapetalk_df[shapetalk_df.changeit_split == SPLIT.TEST.value]
    test_df = test_df[
        test_df[ID].apply(
            lambda row_id: not os.path.exists(
                os.path.join(results_images_dir, f"{row_id}_{copy_steps_list[-1]}.png")
            )
        )
    ]

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
        parts, indices, prompts, source_latents, target_latents = (
            batch[PARTS],
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
        for row_id, pc in zip(indices, pcs):
            pc.save(os.path.join(results_pcs_dir, f"{row_id}_{N_PTS_SAMPLE}.npz"))
            pc.render(os.path.join(results_images_dir, f"{row_id}.png"))

        if os.path.exists(injection_dir):
            shutil.rmtree(injection_dir)
        os.makedirs(injection_dir, exist_ok=True)

        samples_list = model.sampler.sample_batch(
            return_list=True,
            injection_dir=injection_dir,
            batch_size=config[BATCH_SIZE],
            guidances=[target_latents, None],
            model_kwargs={TEXTS: [config[COPY_PROMPT]] * config[BATCH_SIZE]},
        )
        pcs_1024 = model.sampler.output_to_point_clouds(samples_list[0])
        pcs_4096 = model.sampler.output_to_point_clouds(samples_list[1])
        injection_indices_list = []
        for row_id, pc_1024, pc_4096, part in zip(indices, pcs_1024, pcs_4096, parts):
            pc_4096.save(
                os.path.join(results_pcs_dir, f"{row_id}_copy_{N_PTS_SAMPLE}.npz")
            )
            pc_4096.render(os.path.join(results_images_dir, f"{row_id}_copy.png"))
            pc_1024.shape_category = shape_category
            pc_1024 = pc_1024.segment(pointnet)
            injection_indices_list.append(pc_1024.injection_indices(part).to(device))

        for copy_steps in copy_steps_list:
            samples = model.sampler.sample_batch(
                copy_steps=copy_steps,
                injection_dir=injection_dir,
                batch_size=config[BATCH_SIZE],
                guidances=[target_latents, None],
                injection_indices_list=injection_indices_list,
                model_kwargs={TEXTS: [config[COPY_PROMPT]] * config[BATCH_SIZE]},
            )
            pcs = model.sampler.output_to_point_clouds(samples)
            for row_id, pc in zip(indices, pcs):
                pc.save(
                    os.path.join(
                        results_pcs_dir, f"{row_id}_{copy_steps}_{N_PTS_SAMPLE}.npz"
                    )
                )
                pc.render(
                    os.path.join(results_images_dir, f"{row_id}_{copy_steps}.png")
                )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    main(args, device)
