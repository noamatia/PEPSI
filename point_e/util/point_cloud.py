import torch
import random
import numpy as np
import mitsuba as mi
from .ply_util import write_ply
from dataclasses import dataclass
from pepsi.consts import XMLTemplates
from pepsi.custom_types import shape_category_type
from pepsi.pointnet import POINTNET_MODELS, PointNet
from typing import BinaryIO, Dict, List, Optional, Union

mi.set_variant("scalar_rgb")


COLORS = frozenset(["R", "G", "B", "A"])


def preprocess(data, channel):
    if channel in COLORS:
        return np.round(data * 255.0)
    return data


@dataclass
class PointCloud:
    """
    An array of points sampled on a surface. Each point may have zero or more
    channel attributes.

    :param coords: an [N x 3] array of point coordinates.
    :param channels: a dict mapping names to [N] arrays of channel values.
    """

    coords: np.ndarray
    channels: Dict[str, np.ndarray]
    labels: Optional[np.ndarray] = None
    shape_category: Optional[shape_category_type] = None

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.coords,
            rgb=(
                np.stack([self.channels[x] for x in "RGB"], axis=1)
                if all(x in self.channels for x in "RGB")
                else None
            ),
        )

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx
            cur_dists = np.minimum(cur_dists, compute_dists(idx))
        return self.subsample(indices, **subsample_kwargs)

    def subsample(
        self, indices: np.ndarray, average_neighbors: bool = False
    ) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
                shape_category=self.shape_category,
                labels=self.labels[indices] if self.labels is not None else None,
            )

        new_coords = self.coords[indices]
        new_labels = self.labels[indices] if self.labels is not None else None
        neighbor_indices = PointCloud(
            coords=new_coords, channels={}, shape_category=self.shape_category
        ).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(
            coords=new_coords,
            channels=new_channels,
            labels=new_labels,
            shape_category=self.shape_category,
        )

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack(
            [preprocess(self.channels[name], name) for name in channel_names], axis=-1
        )
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = (
                norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            )
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        assert self.shape_category == other.shape_category
        assert not (self.labels is None ^ other.labels is None)
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0)
                for k, v in self.channels.items()
            },
            shape_category=self.shape_category,
            labels=(
                np.concatenate([self.labels, other.labels], axis=0)
                if self.labels is not None and other.labels is not None
                else None
            ),
        )

    def encode(self) -> torch.Tensor:
        coords = torch.tensor(self.coords.T, dtype=torch.float32)
        rgb = [(self.channels[x] * 255).astype(np.uint8) for x in "RGB"]
        rgb = [torch.tensor(x, dtype=torch.float32) for x in rgb]
        rgb = torch.stack(rgb, dim=0)
        return torch.cat([coords, rgb], dim=0)

    @classmethod
    def load_shapenet(
        cls, shapenet_dir: str, shapenet_uid: str, shape_category: shape_category_type
    ) -> "PointCloud":
        path = f"{shapenet_dir}/{shapenet_uid}.npz"
        with open(path, "rb") as f:
            coords = np.load(f)["pointcloud"]
        channels = {k: np.zeros_like(coords[:, 0]) for k in ["R", "G", "B"]}
        return PointCloud(
            coords=coords,
            channels=channels,
            shape_category=shape_category,
        )

    def mask(self, part: str, target_fraction: float = 0.01) -> "PointCloud":
        assert self.shape_category is not None, "No shape category"
        assert self.labels is not None, "No labels"

        label = POINTNET_MODELS[self.shape_category].parts[part]
        mask = np.isin(self.labels, label)

        # Ensure at least 1% of the points are selected
        target_count = int(mask.size * target_fraction)
        current_count = np.sum(mask)
        if current_count < target_count:
            needed_count = target_count - current_count
            zero_indices = np.argwhere(mask == 0)
            random_indices = np.random.choice(
                zero_indices.shape[0], size=needed_count, replace=False
            )
            mask[zero_indices[random_indices]] = 1

        new_coords = self.coords.copy()
        new_channels = {k: v.copy() for k, v in self.channels.items()}
        new_labels = self.labels.copy()

        new_coords[mask] = 0
        for k in ["R", "G", "B"]:
            new_channels[k][mask] = 1

        return PointCloud(
            labels=new_labels,
            coords=new_coords,
            channels=new_channels,
            shape_category=self.shape_category,
        )

    @staticmethod
    def compute_xml_color(x: float, y: float, z: float) -> np.ndarray:
        vec = np.clip(np.array([x, y, z]), 0.001, 1.0)
        vec /= np.linalg.norm(vec)
        return vec

    def generate_xml_content(self, pcl: np.ndarray) -> str:
        xml_segments = [XMLTemplates.HEAD]
        for point in pcl:
            color = self.compute_xml_color(
                point[0] + 0.5, point[1] + 0.5, point[2] + 0.5 - 0.0125
            )
            xml_segments.append(
                XMLTemplates.BALL_SEGMENT.format(point[0], point[1], point[2], *color)
            )
        xml_segments.append(XMLTemplates.TAIL)
        return "".join(xml_segments)

    @staticmethod
    def render_scene(xml_content: str) -> mi.TensorXf:
        scene = mi.load_string(xml_content)
        img = mi.render(scene)
        return img

    @staticmethod
    def save_scene(output_file_path: str, rendered_scene: mi.TensorXf):
        mi.util.write_bitmap(output_file_path, rendered_scene)

    @staticmethod
    def rendered_scene_to_img(rendered_scene: mi.TensorXf) -> np.ndarray:
        img = np.array(rendered_scene)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    def render(
        self, output_file_path: Optional[str] = None, num_points: int = 4096
    ) -> np.ndarray:
        pcl = self.coords[
            np.random.choice(self.coords.shape[0], num_points, replace=False)
        ]
        pcl = pcl[:, [2, 0, 1]]
        pcl[:, 0] *= -1
        pcl[:, 2] += 0.0125
        pcl *= 1.5
        pcl = pcl.astype(np.float32)

        xml_content = self.generate_xml_content(pcl)
        rendered_scene = self.render_scene(xml_content)
        if output_file_path is not None:
            self.save_scene(output_file_path, rendered_scene)
        return self.rendered_scene_to_img(rendered_scene)

    def segment(self, pointnet: PointNet) -> "PointCloud":
        assert self.shape_category == pointnet.shape_category, "Invalid shape category"
        points = self.coords.copy()
        points = points - np.expand_dims(np.mean(points, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(points**2, axis=1)), 0)
        points = points / dist
        points = points.transpose(1, 0)
        points = np.expand_dims(points, axis=0)

        pointnet.net.set_input_shape(points.shape)
        pred, _ = pointnet.net.predict({"point": points})
        labels = np.argmax(pred[0], axis=1)
        values = list(POINTNET_MODELS[self.shape_category].parts.values())
        assert np.isin(labels, np.array(values)).all(), "Invalid part label"

        new_coords = self.coords.copy()
        new_channels = {k: v.copy() for k, v in self.channels.items()}

        return PointCloud(
            labels=labels,
            coords=new_coords,
            channels=new_channels,
            shape_category=self.shape_category,
        )
