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
    ):
        super().__init__()
        self.split = split.value
        self.uid_key = uid_key.value
        self.uids = []
        self.parts = []
        self.prompts = []
        self.source_latents = []
        self.target_latents = []
        self._append_samples(
            pcs_dir, df, shapenet_dir, pointnet, device, uid_key, utterance_key
        )
        self.set_length(batch_size)

    def _append_samples(
        self,
        pcs_dir: str,
        df: pd.DataFrame,
        shapenet_dir: str,
        pointnet: PointNet,
        device: torch.device,
        uid_key: uid_key_type,
        utterance_key: utterance_key_type,
    ):
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Loading data"):
            part = row.part
            shapenet_uid = row[uid_key.value]
            self.uids.append(shapenet_uid)
            self.parts.append(part)
            self.prompts.append(row[utterance_key.value])
            target_pc = PointCloud.load_shapenet(
                shapenet_dir, shapenet_uid, pointnet.shape_category
            )
            target_pc = target_pc.farthest_point_sample(N_PTS_SAMPLE)
            source_pc_file = shapenet_uid.replace("/", "_")
            source_pc_file = f"{source_pc_file}_{part}_{N_PTS_SAMPLE}.npz"
            source_pc_path = os.path.join(pcs_dir, source_pc_file)
            if os.path.exists(source_pc_path):
                source_pc = PointCloud.load(source_pc_path)
            else:
                source_pc = target_pc.segment(pointnet)
                source_pc = source_pc.mask(part)
                source_pc.save(source_pc_path)
            self.source_latents.append(
                source_pc.farthest_point_sample(N_PTS_ENCODE).encode().to(device)
            )
            self.target_latents.append(
                target_pc.farthest_point_sample(N_PTS_ENCODE).encode().to(device)
            )

    def set_length(self, batch_size: int, length: Optional[int] = None):
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
            UID_KEYS: self.uid_key,
            UIDS: self.uids[index],
            PARTS: self.parts[index],
            PROMPTS: self.prompts[index],
            SOURCE_LATENTS: self.source_latents[index],
            TARGET_LATENTS: self.target_latents[index],
        }
