import torch
from torch import nn


class Homography(nn.Module):

    def __init__(self, depths):
      super().__init__()
      self.depths = depths

    def forward(self, images, intrinsics, extrinsics):
        """
        :param torch.Tensor images: (N, T, C, H, W)
        :param torch.Tensor intrinsic: (N, T, 3, 3)
        :param torch.Tensor extrinsic: (N, T, 4, 4)

        :return warped_voxels: (N, T, D, C, H, W) 
        """
        # (N, T, D, 3, 3)
        homographies = self.get_homographies(intrinsics, extrinsics)
        # (N, T, C, HxW, 3)
        coords = self.images_to_coords(images)
        # (N, T, D, HxW, 2)
        warped_coords = self.warp(coords, homographies)
        D = len(self.depths)
        N, T, C, H, W = images.shape
        output_images = []
        for i in range(D):
            output_image = torch.nn.functional.grid_sample(
                input=images.reshape((N*T, C, H, W)),
                grid=warped_coords.reshape((N*T, D, H, W, 2))[:, i, :, :, :],
                mode='bilinear', padding_mode='zeros'
            )
            output_images.append(output_image)
        output_images = torch.stack(output_images, dim=2).reshape((N, T, C, D, H, W))
        return output_images

    def get_homographies(self, intrinsic, extrinsic):
        """
        :param torch.Tensor intrinsic: (N, T, 3, 3)
        :param torch.Tensor extrinsic: (N, T, 4, 4)
        """
        # Parameters
        assert intrinsic.dim() == 4
        assert intrinsic.shape[-2:] == (3,3)
        assert intrinsic.dim() == 4
        assert extrinsic.shape[-2:] == (4,4)
        ## Reference
        K_ref = intrinsic[:, 0] # (N, 3, 3)
        R_ref = extrinsic[:, 0, :3, :3] # (N, 3, 3)
        t_ref = extrinsic[:, 0, :3, 3] # (N, 3)
        n_ref = extrinsic[:, 0, 2, :3] # (N, 3)
        ## Views
        K_view = intrinsic # (N, T, 3, 3)
        R_view = extrinsic[:, :, :3, :3] # (N, T, 3, 3)
        t_view = extrinsic[:, :, :3, 3] # (N, T, 3)
        ## Sizes
        N, T = K_view.shape[:2]
        D = len(self.depths)

        # Vector to matrix for matrix multiplication
        t_ref = t_ref.unsqueeze(dim=2) # (N, 3) -> (N, 3, 1)
        n_ref = n_ref.unsqueeze(dim=2) # (N, 3) -> (N, 3, 1)
        t_view = t_view.unsqueeze(dim=3) # (N, T, 3) -> (N, T, 3, 1)

        # Transition difference
        # [(N, 1, 3, 3) @ (N, 1, 3, 1) - (N, T, 3, 3) @ (N, T, 3, 1)] @ (N, 1, 3) = (N, T, 3, 3)
        t_diff = (
           R_ref.reshape((N, 1, 3, 3)).transpose(-1, -2) @ t_ref.reshape((N, 1, 3, 1))
           - R_view.transpose(-1, -2) @ t_view
        ) @ n_ref.reshape((N, 1, 3, 1)).transpose(-1, -2)

        # Homographies
        t_diff = t_diff.unsqueeze(dim=2).expand((-1, -1, D, -1, -1)) # (N, T, D, 3, 3)
        eye = torch.eye(3).reshape((1, 1, 1, 3, 3)).expand((N, T, D, -1, -1)) # (N, T, D, 3, 3)
        depths = self.depths.reshape((1, 1, -1, 1, 1)).expand((N, T, -1, 3, 3)) # (N, T, D, 3, 3)
        H = eye - t_diff / depths # (N, T, D, 3, 3)

        # From view coordinate system to world coordinate system
        K_view = K_view.reshape(N, T, 1, 3, 3).expand((-1, -1, D, -1, -1)) # (N, T, D, 3, 3)
        R_view = R_view.reshape(N, T, 1, 3, 3).expand((-1, -1, D, -1, -1)) # (N, T, D, 3, 3)
        world = K_view @ R_view @ H
        
        reference = (R_ref @ K_ref.inverse()).reshape(N, 1, 1, 3, 3).expand((-1, T, D, -1, -1)) # (N, T, D, 3, 3)
        homographies = world @ reference # (N, T, D, 3, 3)
        
        return homographies
    
    def images_to_coords(self, images:torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor images: (N, T, C, H, W)
        :return torch.Tensor coords: (N, T, C, H x W, 3)
        """
        N, T, C, H, W = images.shape
        x_coords = torch.arange(0.5, W + 0.5, 1)
        y_coords = torch.arange(0.5, H + 0.5, 1)
        z_coords = torch.tensor([1.0])
        coords = torch.cartesian_prod(x_coords, y_coords, z_coords)
        return coords
    
    def warp(self, coords, homographies):
        """
        :param torch.Tensor homographies: (N, T, D, 3, 3)
        :param torch.Tensor coords: (HxW, 3)
        :return warped_coords: (N, T, D, HxW, 2)
        """
        N, T, D = homographies.shape[:3]        
        homographies = homographies.reshape((N, T, D, 1, 1, 3, 3)) # (N, T, D, 1, 3, 3)
        coords = coords.unsqueeze(dim=2) # (HxW, 3, 1)
        new_coords = homographies @ coords # (N, T, D, HxW, 3, 1)
        new_coords = new_coords.squeeze(dim=-1) # (N, T, D, HxW, 3)
        xy_coords = new_coords[..., :2]
        z_coords = new_coords[..., 2]
        del new_coords
        z_coords[z_coords == 0] = 1e-7
        xy_coords /= z_coords.unsqueeze(dim=5) # (N, T, D, HxW, 2)
        return xy_coords
    
    def interpolate(self, images, coords):
        """
        :param torch.Tensor images: (N, T, C, H, W)
        :param torch.Tensor coords: (N, T, D, HxW, 3)
        """
        N, T, C, H, W = images.shape
        D = coords.shape[2]

        # Split coordinates
        x = coords[..., 0] - 0.5  # shape: (N, T, D, H*W)
        y = coords[..., 1] - 0.5  # shape: (N, T, D, H*W)

        # Neighboring pixels
        x0 = torch.floor(x).long()     # Left
        x1 = x0 + 1                    # Right
        y0 = torch.floor(y).long()     # Bottom
        y1 = y0 + 1                    # Top

        # Clamp: restrict to inside the image
        x0 = x0.clamp(0, W - 1)
        x1 = x1.clamp(0, W - 1)
        y0 = y0.clamp(0, H - 1)
        y1 = y1.clamp(0, H - 1)

        def gather(flat_img, ix, iy):
            """
            img: (N, T, C, HW)
            ix, iy: (N, T, D, HW)
            Returns: (N, T, C, HW)
            """
            flat_img = flat_img.reshape((N, T, C, H*W))
            idx = iy * W + ix  # flatten 2D index
            idx = idx.reshape((N, T, 1, D*H*W)).expand((-1, -1, C, -1))
            pixels = torch.gather(flat_img, dim=-1, index=idx)
            pixels = pixels.reshape((N, T, C, D, H*W))

        # Gather pixels
        images = images.reshape(N, T, C, H * W)
        Ia = gather(images, x0, y0)
        Ib = gather(images, x1, y0)
        Ic = gather(images, x0, y1)
        Id = gather(images, x1, y1)

        # Interpolation weights
        x0f, x1f = x0.float(), x1.float()
        y0f, y1f = y0.float(), y1.float()

        wa = (x1f - x) * (y1f - y)
        wb = (x - x0f) * (y1f - y)
        wc = (x1f - x) * (y - y0f)
        wd = (x - x0f) * (y - y0f)

        # Weighted sum
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id  # shape: (N, T, C, D, HW)
        out = out.reshape((N, T, C, D, H, W))
        return out