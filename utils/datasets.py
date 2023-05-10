import numpy as np
import torch
import torch.cuda
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize


class PointCloudDataset:
    def __init__(
        self,
        root,
        voxel_size: float = 0.1,
        label: int = 0,
        initial_fov: float = 60,
        initial_scale: float = 1,
        use_uv: bool = True,
        distort_shift: bool = True,
        distort_focal: bool = True,
    ):
        self.depths_dir = root / "depths"
        self.depths_names = sorted(list(self.depths_dir.glob("*.npz")))
        self.intrinsics_dir = root / "intrinsics"
        self.label = label
        self.intial_fov = initial_fov
        self.intial_scale = initial_scale
        self.distort_shift = distort_shift
        self.distort_focal = distort_focal
        self.use_uv = use_uv
        self.intrinsics = []

        for p in self.depths_names:
            intrinsic_path = self.intrinsics_dir / f"{p.name[:-4]}.npy"
            self.intrinsics.append(np.load(intrinsic_path))

        self.voxel_size = voxel_size

    def __len__(self):
        return len(self.depths_names)

    def __getitem__(self, idx: int):
        """this function is used for dataloading, it load raw_data alongside its correspandant intrinsics matrix,
         necessary perturbation requested by the user will be applied on the fly,and finally it create SparseTensors from raw_data and model's output tensors for the labels

        Args:
            idx: index of the trainig example to process

        """

        depth_path = self.depths_dir / self.depths_names[idx]
        depth = np.load(depth_path)["arr_0"]

        h, w = depth.shape
        valid_mask = depth > 0
        depth = depth[valid_mask]

        depth_shifted_normalized, shift, scale = self.normalize_and_shift(
            depth, shift=self.distort_shift
        )

        u, v = self.init_image_coords(h, w)
        uv = np.vstack([u[valid_mask], v[valid_mask]]).T

        intrinsics = self.intrinsics[idx]
        real_focal_length = (intrinsics[0, 0] + intrinsics[1, 1]) / 2
        initial_scale = self.intial_scale

        if self.distort_focal:
            initial_focal_length = (
                h // 2 / np.tan((self.intial_fov / 2.0) * np.pi / 180)
            )

        else:
            initial_focal_length = real_focal_length

        jammed_intrinsics = intrinsics.copy()
        jammed_intrinsics[0, 0] = initial_focal_length
        jammed_intrinsics[1, 1] = initial_focal_length

        delta_d = shift
        alpha_f = real_focal_length / initial_focal_length
        beta_s = scale / initial_scale / 50

        labels = [delta_d, beta_s, alpha_f]

        xyz = initial_scale * self.unproject_points(
            uv, jammed_intrinsics, depth_shifted_normalized
        )

        coords = xyz[:, :3]

        if self.use_uv:
            uv = (uv - intrinsics[:2, 2][np.newaxis, ...]) / initial_focal_length
            xyzuv = np.hstack([xyz[:, :3], uv])
            feats = xyzuv
        else:
            feats = xyz

        label = torch.tensor(labels[self.label], dtype=torch.float)

        coords -= np.min(coords, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords, self.voxel_size, return_index=True)

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        input = SparseTensor(coords=coords, feats=feats)

        return {"input": input, "label": label}

    def unproject_points(
        self, pcd_uv: np.ndarray, intrinsics_matrix: np.ndarray, d: np.ndarray
    ) -> np.ndarray:
        """unproject depth values to the 3D space

               (u-u0)                  (v-v0)
          x =  ------ d         y =    ------ d         z = d
                 fx                      fy

        Args:
            pcd_uv (_list_): a list of the image cordinates of each pixels
            intrinsics_matrix (_ndarray_): a 3 by 3 matrix of camera intrensic parameters
            d (_list_): a list of depth values of each pixel

        Returns:
            _ndarray_: an array of n*3 of unprojected coordiantes of each pixels
        """
        pcduv1 = np.hstack([pcd_uv, np.ones((pcd_uv.shape[0], 1))])

        uv0 = np.asarray([intrinsics_matrix[0, 2], intrinsics_matrix[1, 2], 0])[
            np.newaxis, ...
        ]
        f = np.asarray([intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], 1])[
            np.newaxis, ...
        ]
        unprojectedPCD = (pcduv1 - uv0) * d[..., np.newaxis] / f

        return unprojectedPCD

    def init_image_coords(
        self, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """generating pixels cordinates in pixels

        Args:
            height (int): height of the rgb/depth images
            width (int): width of the rgb/depth images

        Returns:
            ndarray: h by w grids representing the coordinates each pixels in the camera plane
        """
        x_row = np.arange(0, width)
        x = np.tile(x_row, (height, 1))
        x = x.astype(np.float32)
        u_u0 = x

        y_col = np.arange(0, height)
        y = np.tile(y_col, (width, 1)).T
        y = y.astype(np.float32)
        v_v0 = y

        return u_u0, v_v0

    def normalize_and_shift(
        self, gt_depths: np.ndarray, shift: bool = True, initial_min_value: float = 0.51
    ) -> tuple[np.ndarray, float, float]:
        """Normalize depth values to a unitary scale and adding a tracked shift to the data which will be used as a label

        Args:
            gt_depths (list): a list holding the GT depth values for all valid pixels
            initial_min_value (float, optional): the initiale mimimum value of the shifted and depths. Defaults to 0.51.

        Returns:
            depths (list): a list of normlized and shifted depth values
            shift (float): the applied shift to the depth values during the shifting & normalization process
            scale (float): the correct scale of the normalized depths
            these varibles allows writing : gt_depths = scale * (depths + shift)/(1 + shift)

        """
        trimmed_gt_depths = gt_depths[
            (gt_depths > np.percentile(gt_depths, 10))
            & (gt_depths < np.percentile(gt_depths, 90))
        ]
        mu = trimmed_gt_depths.mean()
        sigma = trimmed_gt_depths.std()
        shift = 0
        scale = sigma
        if shift:
            depths = (gt_depths - mu) / sigma
            shift += mu / sigma
            min_depths = depths.min()
            depths -= min_depths
            shift += min_depths
            depths += initial_min_value
            shift -= initial_min_value
        else:
            depths = gt_depths / sigma

        max_depths = depths.max()
        depths /= max_depths
        shift /= max_depths
        scale *= max_depths

        return depths, shift, scale
