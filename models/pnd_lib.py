import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def build_non_linearity(non_linearity_type, num_features):
    return non_linearity_type()


class Conv2(nn.Module):
    def __init__(self, in_features, hid_features, out_features, norm_type=nn.BatchNorm2d, non_linearity_type=nn.ReLU,
                 groups=1, conv_type=nn.Conv2d, kernel_size=3, stride=1):
        super(Conv2, self).__init__()
        self.conv1 = conv_type(in_channels=in_features, out_channels=hid_features, kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               groups=groups)
        self.norm1 = norm_type(hid_features)
        self.non_linear1 = build_non_linearity(non_linearity_type, hid_features)
        self.conv2 = nn.Conv2d(in_channels=hid_features, out_channels=out_features, kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.non_linear1(x)
        x = self.conv2(x)
        return x


class SimpleGate(nn.Module):
    def __init__(self, in_dims, hid_dims=16, output_dims=10, non_linearity_type=nn.ReLU, norm_type=nn.BatchNorm1d):
        super(SimpleGate, self).__init__()
        self.net = nn.Sequential(
            self.get_linearity_type()(in_dims, hid_dims),
            norm_type(hid_dims),
            build_non_linearity(non_linearity_type, hid_dims),
            nn.Linear(hid_dims, output_dims)
        )

    def get_linearity_type(self):
        return nn.Linear

    def forward(self, x):
        if len(x.shape) > 2:
            x = F.adaptive_avg_pool2d(x, 1).squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.net(x)
        x = torch.sigmoid(x) #.squeeze()
        return x



class ExitModule(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims, cam_hid_dims=None,
                 scale_factor=1,
                 groups=1,
                 kernel_size=3,
                 stride=None,
                 initial_conv_type=Conv2,
                 conv_bias=False,
                 conv_type=nn.Conv2d,
                 norm_type=nn.BatchNorm2d,
                 non_linearity_type=nn.ReLU,
                 gate_type=SimpleGate,
                 gate_norm_type=nn.BatchNorm1d,
                 gate_non_linearity_type=nn.ReLU,
                 ):
        super(ExitModule, self).__init__()
        self.in_dims = in_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        if cam_hid_dims is None:
            cam_hid_dims = self.hid_dims
        self.cam_hid_dims = cam_hid_dims
        self.initial_conv_type = initial_conv_type
        self.conv_bias = conv_bias
        self.conv_type = conv_type
        self.scale_factor = scale_factor
        self.groups = groups
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size // 2
        self.stride = stride
        self.norm_type = norm_type
        self.non_linearity_type = non_linearity_type
        self.gate_type = gate_type
        self.gate_norm_type = gate_norm_type
        self.gate_non_linearity_type = gate_non_linearity_type
        self.build_network()

    def build_network(self):
        self.initial_convs_i = nn.Sequential(self.initial_conv_type(self.in_dims,
                                            self.hid_dims,
                                            self.cam_hid_dims,
                                            norm_type=self.norm_type,
                                            non_linearity_type=self.non_linearity_type,
                                            conv_type=self.conv_type,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride), build_non_linearity(self.non_linearity_type, self.cam_hid_dims))
        self.initial_convs_b = nn.Sequential(self.initial_conv_type(self.in_dims,
                                            self.hid_dims,
                                            self.cam_hid_dims,
                                            norm_type=self.norm_type,
                                            non_linearity_type=self.non_linearity_type,
                                            conv_type=self.conv_type,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride), build_non_linearity(self.non_linearity_type, self.cam_hid_dims))

        self.dm_i = self.gate_type(self.cam_hid_dims*2,output_dims = self.out_dims,
                                       norm_type=self.gate_norm_type,
                                       non_linearity_type=self.gate_non_linearity_type)

        self.dm_b = self.gate_type(self.cam_hid_dims*2,output_dims = self.out_dims,
                                       norm_type=self.gate_norm_type,
                                       non_linearity_type=self.gate_non_linearity_type)

    def forward(self, x_i, x_b, y, use_mix = False):
        out = {}

        if self.scale_factor != 1:
            x_i = F.interpolate(x_i, scale_factor=self.scale_factor, align_corners=False, mode='bilinear')
            x_b = F.interpolate(x_b, scale_factor=self.scale_factor, align_corners=False, mode='bilinear')

        x_i = self.initial_convs_i(x_i)
        x_b = self.initial_convs_b(x_b)
   
        x_conflict = torch.cat((x_i, x_b.detach()), dim=1)
        x_align = torch.cat((x_i.detach(), x_b), dim=1)
        
        dm_out_conflict = self.dm_i(x_conflict)
        dm_out_align = self.dm_b(x_align)
        out['dm_conflict_out'] = dm_out_conflict
        out['dm_align_out'] = dm_out_align
         

        if use_mix:
            indices_mini = np.random.choice(x_i.size(0),16,replace=False)
            indices_mini_swap = np.random.permutation(indices_mini)
            indices_mini_aug_swap = np.random.permutation(x_i.size(0))
            x_i_mini = x_i[indices_mini]   
            x_b_mini = x_b[indices_mini]   
            x_b_mini_swap = x_b[indices_mini_swap] 
            x_b_mini_aug = x_b_mini.unsqueeze(1)
            x_b_mini_aug = x_b_mini_aug.repeat(1, int(x_i.size(0)/16),1,1,1)  
            x_b_mini_aug = x_b_mini_aug.view(-1,1,x_b_mini_aug.size(2),x_b_mini_aug.size(3),x_b_mini_aug.size(4)).squeeze(1)
            x_i_mini_aug_swap = x_i[indices_mini_aug_swap]        
            x_pos = torch.cat((x_i_mini, x_b_mini_swap.detach()), dim=1)
            x_neg = torch.cat((x_i_mini_aug_swap.detach(), x_b_mini_aug), dim=1)
  
            x_pos_pred = self.dm_i(x_pos)
            x_neg_pred = self.dm_b(x_neg) 

            out['dm_out_mix'] = [x_pos_pred,x_neg_pred]
            out['indices_mini'] = indices_mini           
        return out

        


class MultiExitModule(nn.Module):
    """
    Holds multiple exits
    It passes intermediate representations through those exits to gather CAMs/predictions
    """

    def __init__(
            self,
            detached_exit_ixs=[0],
            exit_out_dims=None,
            exit_block_nums=[0, 1, 2, 3],
            exit_type=ExitModule,
            exit_gate_type=SimpleGate,
            exit_initial_conv_type=Conv2,
            exit_hid_dims=[None, None, None, None],
            exit_width_factors=[2, 1, 1 / 2, 1 / 4],
            cam_width_factors=[8, 4, 2, 1],
            exit_scale_factors=[1, 1, 1, 1],
            exit_kernel_sizes=[3, 3, 3, 3],
            exit_strides=[None] * 4,
            inference_earliest_exit_ix=1,
            downsample_factors_for_scores=[1 / 8, 1 / 4, 1 / 2, 1],
    ) -> None:
        """
        Adds multiple exits to DenseNet
        :param detached_exit_ixs: Exit ixs whose gradients should not flow into the trunk
        :param exit_out_dims: e.g., # of classes
        :param exit_block_nums: Blocks where the exits are attached (EfficientNets have 9 blocks (0-8))
        :param exit_type: Class of the exit that performs predictions
        :param exit_gate_type: Class of exit gate that decides whether or not to terminate a sample
        :param exit_initial_conv_type: Initial layer of the exit
        :param exit_width_factors:
        :param cam_width_factors:
        :param exit_scale_factors:
        :param inference_earliest_exit_ix: The first exit to use for inference (default=1 i.e., E.0 is not used for inference)

        """
        super().__init__()
        self.detached_exit_ixs = detached_exit_ixs
        self.exit_out_dims = exit_out_dims
        self.exit_block_nums = exit_block_nums
        self.exit_type = exit_type
        self.exit_gate_type = exit_gate_type
        self.exit_initial_conv_type = exit_initial_conv_type
        self.exit_hid_dims = exit_hid_dims
        self.exit_width_factors = exit_width_factors
        self.cam_width_factors = cam_width_factors
        self.exit_scale_factors = exit_scale_factors
        self.exit_kernel_sizes = exit_kernel_sizes
        self.exit_strides = exit_strides
        self.inference_earliest_exit_ix = inference_earliest_exit_ix
        self.downsample_factors_for_scores = downsample_factors_for_scores
        self.exits = []

    def build_and_add_exit(self, in_dims):
        exit_ix = len(self.exits)
        _hid_dims = self.exit_hid_dims[exit_ix]
        if _hid_dims is None:
            _hid_dims = int(in_dims * self.exit_width_factors[exit_ix])
        exit = self.exit_type(
            in_dims=in_dims,
            out_dims=self.exit_out_dims,
            hid_dims=_hid_dims,
            cam_hid_dims=int(in_dims * self.cam_width_factors[exit_ix]),
            kernel_size=self.exit_kernel_sizes[exit_ix],
            stride=self.exit_strides[exit_ix],
            scale_factor=self.exit_scale_factors[exit_ix],
        )
        if hasattr(exit, 'set_downsample_factor'):
            exit.set_downsample_factor(self.downsample_factors_for_scores[exit_ix])
        self.exits.append(exit)
        self.exits = nn.ModuleList(self.exits)



    def get_exit_block_nums(self):
        return self.exit_block_nums


    def forward(self,block_num_to_exit_in_i, block_num_to_exit_in_b, y, use_mix = False, exit_strategy=None):
        exit_outs = {}
        exit_ix = 0
        for block_num in block_num_to_exit_in_i:
            if block_num in self.exit_block_nums:
                exit_in_i = block_num_to_exit_in_i[block_num]
                exit_in_b = block_num_to_exit_in_b[block_num]
                exit_out = self.exits[exit_ix](exit_in_i,exit_in_b,y, use_mix = use_mix)
                for k in exit_out:
                    exit_outs[f"E={exit_ix}, {k}"] = exit_out[k]
                exit_ix += 1      
        return exit_outs



