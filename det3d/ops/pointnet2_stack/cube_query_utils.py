import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from . import pointnet2_stack_cuda as pointnet2
from . import pointnet2_utils


class CubeQuery(Function):

    @staticmethod
    def forward(ctx, radius_x: float, radius_y: float, radius_z: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt):
        """
        Args:
            ctx:
            radius: float, radius of the balls
            nsample: int, maximum number of features in the balls
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
            empty_ball_mask: (M1 + M2, )
            non_padding: (M1 + M2, nsample) tensor indicating sample_i is padding (denotes 0) or not (denotes 1)
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(M, nsample).zero_()
        non_padding = torch.cuda.FloatTensor(M, nsample).zero_()

        pointnet2.cube_query_wrapper(
            B, M, 
            radius_x, radius_y, radius_z, 
            nsample, 
            new_xyz, new_xyz_batch_cnt, 
            xyz, xyz_batch_cnt, 
            idx, non_padding
        )

        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask, non_padding

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


cube_query = CubeQuery.apply


class CubeQueryAndGroup(nn.Module):
    def __init__(self, radius_x: float, radius_y: float, radius_z: float, nsample: int, use_xyz: bool = False):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius_x, self.radius_y, self.radius_z = radius_x, radius_y, radius_z
        self.nsample, self.use_xyz = nsample, use_xyz

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                features: torch.Tensor = None):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
            num_non_padding: (M1 + M2, ) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))
        assert xyz.shape[1] == new_xyz.shape[1], 'xyz: %s, new_xyz: %s' % (str(xyz.shape), str(new_xyz.shape))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...), non_padding: (M1 + M2 ..., nsample)
        # non_padding_mask: non_padding部分是1, padding部分是0
        idx, empty_ball_mask, non_padding = cube_query(
            self.radius_x,
            self.radius_y,
            self.radius_z, 
            self.nsample, 
            xyz, 
            xyz_batch_cnt, 
            new_xyz, 
            new_xyz_batch_cnt
        )

        # 先grouping feature，再根据padding_mask置零 
        # grouped_features: (M1 + M2, C, nsample)
        grouped_features = pointnet2_utils.grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  
        grouped_features[empty_ball_mask] = 0
        padding_mask = (non_padding < 1)
        # print("==> grouped_features.shape: ", grouped_features.shape)
        # print("==> 0. padding_mask.shape: ", padding_mask.shape)

        padding_mask = padding_mask[:, None, :].expand_as(grouped_features)
        # print("==> 1. padding_mask.shape: ", padding_mask.shape)
        # print("==> 1. padding_mask[0, :, :]: ", padding_mask[0, :, :])
        # print("==> 1. padding_mask[10, :, :]: ", padding_mask[10, :, :])

        grouped_features[padding_mask] = 0
        num_non_padding = torch.sum(non_padding, dim=-1)

        return grouped_features, num_non_padding



if __name__ == '__main__':
    pass
