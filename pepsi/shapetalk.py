import os
import tqdm
import torch
import pandas as pd
from pepsi.consts import *
from typing import Optional
from pepsi.custom_types import *
from torch.utils.data import Dataset
from point_e.util.point_cloud import PointCloud, PointNet


class ShapeTalk(Dataset):
    def __init__(
        self,
        pcs_dir: str,
        batch_size: int,
        df: pd.DataFrame,
        split: split_type,
        shapenet_dir: str,
        pointnet: PointNet,
        device: torch.device,
        uid_key: uid_key_type,
        utterance_key: utterance_key_type,
        images_dir: Optional[str] = None,
    ):
        super().__init__()
        self.split = split.value
        self.parts = []
        self.prompts = []
        self.indices = []
        self.source_latents = []
        self.target_latents = []
        self._append_samples(
            df=df,
            device=device,
            uid_key=uid_key,
            pcs_dir=pcs_dir,
            pointnet=pointnet,
            images_dir=images_dir,
            shapenet_dir=shapenet_dir,
            utterance_key=utterance_key,
        )
        self._set_length(batch_size)

    def _append_samples(
        self,
        pcs_dir: str,
        df: pd.DataFrame,
        shapenet_dir: str,
        pointnet: PointNet,
        device: torch.device,
        uid_key: uid_key_type,
        images_dir: Optional[str],
        utterance_key: utterance_key_type,
    ):
        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Loading data"):
            self.parts.append(row.part)
            self.indices.append(idx)
            self.prompts.append(row[utterance_key.value])
            file_prefix = row[uid_key.value].replace("/", "_")

            target_pc_file = f"{file_prefix}_{N_PTS_SAMPLE}.npz"
            target_pc_path = os.path.join(pcs_dir, target_pc_file)
            if os.path.exists(target_pc_path):
                target_pc = PointCloud.load(target_pc_path)
            else:
                target_pc = PointCloud.load_shapenet(
                    shapenet_dir, row[uid_key.value], pointnet.shape_category
                )
                target_pc = target_pc.farthest_point_sample(N_PTS_SAMPLE)
                target_pc.save(target_pc_path)
            if images_dir is not None:
                target_image_path = os.path.join(images_dir, f"{file_prefix}.png")
                if not os.path.exists(target_image_path):
                    target_pc.render(output_file_path=target_image_path)

            source_pc_file = f"{file_prefix}_{row.part}_{N_PTS_SAMPLE}.npz"
            source_pc_path = os.path.join(pcs_dir, source_pc_file)
            if os.path.exists(source_pc_path):
                source_pc = PointCloud.load(source_pc_path)
            else:
                source_pc = target_pc.segment(pointnet)
                source_pc = source_pc.mask(row.part)
                source_pc.save(source_pc_path)
            if images_dir is not None:
                source_image_path = os.path.join(
                    images_dir, f"{file_prefix}_{row.part}.png"
                )
                if not os.path.exists(source_image_path):
                    source_pc.render(output_file_path=source_image_path)

            self.source_latents.append(
                source_pc.farthest_point_sample(N_PTS_ENCODE).encode().to(device)
            )
            self.target_latents.append(
                target_pc.farthest_point_sample(N_PTS_ENCODE).encode().to(device)
            )

    def _set_length(self, batch_size: int, length: Optional[int] = None):
        if length is None:
            self.length = len(self.prompts)
        else:
            assert length <= len(self.prompts)
            self.length = length
        r = self.length % batch_size
        if r == 0:
            self.logical_length = self.length
        else:
            q = batch_size - r
            self.logical_length = self.length + q

    def __len__(self) -> int:
        return self.logical_length

    def __getitem__(self, logical_index: int) -> Dict:
        index = logical_index % self.length
        return {
            SPLITS: self.split,
            PARTS: self.parts[index],
            PROMPTS: self.prompts[index],
            INDICES: self.indices[index],
            SOURCE_LATENTS: self.source_latents[index],
            TARGET_LATENTS: self.target_latents[index],
        }
