import os
import ssl
import ailia
import shutil
import urllib
from typing import Dict
from pepsi.consts import *
from pepsi.custom_types import *


POINTNET_MODELS: Dict[shape_category_type, PointNetModel] = {
    SHAPE_CATEGORY.CHAIR: PointNetModel(
        weight="chair_100.onnx",
        model="chair_100.onnx.prototxt",
        parts={
            "back": 0,
            "seat": 1,
            "leg": 2,
            "arm": 3,
        },
    ),
    SHAPE_CATEGORY.TABLE: PointNetModel(
        weight="table_100.onnx",
        model="table_100.onnx.prototxt",
        parts={
            "top": 0,
            "leg": 1,
            "support": 2,
        },
    ),
    SHAPE_CATEGORY.LAMP: PointNetModel(
        weight="lamp_100.onnx",
        model="lamp_100.onnx.prototxt",
        parts={
            "base": 0,
            "shade": 1,
            "bulb": 2,
            "tube": 3,
        },
    ),
}


class PointNet:
    def __init__(
        self,
        pointnet_models_dir: str,
        shape_category: shape_category_type,
    ):
        self.shape_category = shape_category
        os.makedirs(pointnet_models_dir, exist_ok=True)
        weight_file = POINTNET_MODELS[shape_category].weight
        weight_path = self.download(pointnet_models_dir, weight_file)
        model_file = POINTNET_MODELS[shape_category].model
        model_path = self.download(pointnet_models_dir, model_file)
        self.net = ailia.Net(model_path, weight_path)

    @classmethod
    def download(cls, destination_dir: str, file_name: str) -> str:
        destination_path = os.path.join(destination_dir, file_name)
        if not os.path.exists(destination_path):
            temp_path = destination_path + ".tmp"
            remote_path = f"{PARTNET_REMOTE_DIR}/{file_name}"
            try:
                urllib.request.urlretrieve(remote_path, temp_path)
            except ssl.SSLError:
                remote_path = remote_path.replace("https", "http")
                urllib.request.urlretrieve(remote_path, temp_path)
            shutil.move(temp_path, destination_path)
        return destination_path
