from itertools import count
import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from . import pointnet2_stack_cuda as pointnet2

class BallQuerWithCount(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, label: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_label: torch.Tensor, new_xyz_batch_cnt: torch.Tensor):
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
        """
        assert new_xyz.is_contiguous()
        assert new_label.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()

        assert xyz.is_contiguous()
        assert label.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        # idx = torch.cuda.IntTensor(M, nsample).zero_()
        count = torch.cuda.IntTensor(M, ).zero_()

        pointnet2.ball_query_with_count_wrapper(
            B, M, radius, nsample, 
            new_xyz, new_label, new_xyz_batch_cnt, 
            xyz, label, xyz_batch_cnt, 
            count)

        # empty_ball_mask = (idx[:, 0] == -1)
        # idx[empty_ball_mask] = 0
        # return idx, empty_ball_mask

        return count

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None, None


ball_query_with_count = BallQuerWithCount.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
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
        """
        assert new_xyz.is_contiguous()
        assert new_xyz_batch_cnt.is_contiguous()
        assert xyz.is_contiguous()
        assert xyz_batch_cnt.is_contiguous()

        B = xyz_batch_cnt.shape[0]
        M = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(M, nsample).zero_()

        pointnet2.ball_query_wrapper(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx)
        empty_ball_mask = (idx[:, 0] == -1)
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample)

        pointnet2.group_points_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                            idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

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
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))

        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        grouped_xyz = grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_xyz -= new_xyz.unsqueeze(-1)

        grouped_xyz[empty_ball_mask] = 0

        if features is not None:
            grouped_features = grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, C, nsample)
            grouped_features[empty_ball_mask] = 0
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (M1 + M2 ..., C + 3, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features, idx


class QueryAndGroupLocalRelation(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
            use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                pred: torch.Tensor, tgt: torch.Tensor, 
                new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor):
        """
        Args:
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query, 暂时用的同一个东西.
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]


            pred: (N1 + N2 ..., C) pred logits
            tgt: (N1 + N2 ..., C) onehot targets

        Returns:
            pred_local_relation: (M1 + M2, C, nsample) tensor
            tgt_local_relation: (M1 + M2, C, nsample) tensor
            idx: (M1 + M2, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_xyz.shape[0] == new_xyz_batch_cnt.sum(), \
            'new_xyz: %s, new_xyz_batch_cnt: %s' % (str(new_xyz.shape), str(new_xyz_batch_cnt))
        assert torch.equal(xyz, new_xyz)
        assert torch.equal(xyz_batch_cnt, new_xyz_batch_cnt)


        # idx: (M1 + M2 ..., nsample), empty_ball_mask: (M1 + M2 ...)
        # empty_ball_mask里面empty的为True, 非空的为False
        idx, empty_ball_mask = ball_query(self.radius, self.nsample, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt)
        
        grouped_pred = grouping_operation(pred, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)
        grouped_tgt = grouping_operation(tgt, xyz_batch_cnt, idx, new_xyz_batch_cnt)  # (M1 + M2, 3, nsample)

        # """
        nonempty_ball_mask = ~empty_ball_mask
        new_grouped_pred = grouped_pred[nonempty_ball_mask]
        new_grouped_tgt = grouped_tgt[nonempty_ball_mask]
        new_grouped_idx = idx[nonempty_ball_mask]
        new_pred = pred[nonempty_ball_mask]
        new_tgt = tgt[nonempty_ball_mask]
        # """

        """
        new_grouped_pred = grouped_pred
        new_grouped_tgt = grouped_tgt
        new_grouped_idx = idx
        new_pred = pred
        new_tgt = tgt
        """


        # local relation
        # (Ni1+Ni2+..., C, nsample) - (Ni1+Ni2+..., C, None) -> (Ni1+Ni2+..., C, nsample)
        pred_local_relation = new_grouped_pred - new_pred[:, :, None]
        tgt_local_relation = new_grouped_tgt - new_tgt[:, :, None]

        
        tgt_local_relation_test = torch.sum((tgt_local_relation != 0), dim=-1)
        tgt_local_relation_test = torch.sum(tgt_local_relation_test, dim=-1)
        # print("==> tgt_local_relation_test[:100]: ", tgt_local_relation_test[:100])

        # 根据 ball_idx 来获得weight
        with torch.no_grad():
            ball_idx_0 = new_grouped_idx[:, 0:1].expand_as(new_grouped_idx)
            eq_flag = torch.eq(new_grouped_idx, ball_idx_0)
            not_eq_flag = ~eq_flag
            # 强行把第0个元素置为True
            # not_eq_flag: (Ni1+Ni2+..., nsample), 对于非重复的idx为True, 重复的idx为False
            not_eq_flag[:, 0] = True
            # 在非重复的位置上取平均.
            a = not_eq_flag.float()
            b = torch.sum(not_eq_flag.float(), dim=-1)[:, None]
            # 分母不会为0.
            weight = a / b
            # print("==> b.flat()[:100]: ", b.flatten()[:100])
            # print("==> torch.max(b): ", torch.max(b))
            # print("==> weight.shape: ", weight.shape)


        return pred_local_relation, tgt_local_relation, weight


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int):
        """
        Args:
            ctx:
            xyz: (B, N, 3) where N > npoint
            npoint: int, number of features in the sampled set

        Returns:
            output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, unknown_batch_cnt, known, known_batch_cnt):
        """
        Args:
            ctx:
            unknown: (N1 + N2..., 3)
            unknown_batch_cnt: (batch_size), [N1, N2, ...]
            known: (M1 + M2..., 3)
            known_batch_cnt: (batch_size), [M1, M2, ...]

        Returns:
            dist: (N1 + N2 ..., 3)  l2 distance to the three nearest neighbors
            idx: (N1 + N2 ..., 3)  index of the three nearest neighbors, range [0, M1+M2+...]
        """
        assert unknown.shape.__len__() == 2 and unknown.shape[1] == 3
        assert known.shape.__len__() == 2 and known.shape[1] == 3
        assert unknown_batch_cnt.__len__() == known_batch_cnt.__len__()

        dist2 = unknown.new_zeros(unknown.shape)
        idx = unknown_batch_cnt.new_zeros(unknown.shape).int()

        pointnet2.three_nn_wrapper(
            unknown.contiguous(), unknown_batch_cnt.contiguous(),
            known.contiguous(), known_batch_cnt.contiguous(), dist2, idx
        )
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor):
        """
        Args:
            ctx:
            features: (M1 + M2 ..., C)
            idx: [N1 + N2 ..., 3]
            weight: [N1 + N2 ..., 3]

        Returns:
            out_tensor: (N1 + N2 ..., C)
        """
        assert idx.shape[0] == weight.shape[0] and idx.shape[1] == weight.shape[1] == 3

        ctx.three_interpolate_for_backward = (idx, weight, features.shape[0])
        output = features.new_zeros((idx.shape[0], features.shape[1]))
        pointnet2.three_interpolate_wrapper(features.contiguous(), idx.contiguous(), weight.contiguous(), output)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (N1 + N2 ..., C)

        Returns:
            grad_features: (M1 + M2 ..., C)
        """
        idx, weight, M = ctx.three_interpolate_for_backward
        grad_features = grad_out.new_zeros((M, grad_out.shape[1]))
        pointnet2.three_interpolate_grad_wrapper(
            grad_out.contiguous(), idx.contiguous(), weight.contiguous(), grad_features
        )
        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


if __name__ == '__main__':
    pass
