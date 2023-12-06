from models.pnd_lib import *
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18



class PnDNet(nn.Module):
    def __init__(self, num_classes=10, multi_exit_type=MultiExitModule,exits_kwargs={}):
        super(PnDNet, self).__init__()
        self.num_classes = num_classes
        self.model_d = nn.Sequential(*list(resnet18(pretrained=True).children()))[:-1]
        self.model_b = nn.Sequential(*list(resnet18(pretrained=True).children()))[:-1]

        exits_kwargs['exit_out_dims'] = num_classes
        out_dims = [64, 128, 256, 512]
        self.exits_cfg = exits_kwargs
        multi_exit = multi_exit_type(**exits_kwargs)
        for i in range(0, 4):
            multi_exit.build_and_add_exit(out_dims[i])
        self.multi_exit = multi_exit

        self.gate_fc = nn.Linear(self.exits_cfg['exit_out_dims']*4, 4)


    def forward(self, x, y = None, use_mix = False):
        block_num_to_exit_in_i = {}
        block_num_to_exit_in_b = {}
        x_d = self.model_d[:4](x)
        for i in range(0, 4):
            x_d = self.model_d[4+i](x_d)
            block_num_to_exit_in_i[i] = x_d     

        ##### bias network
        x_b = self.model_b[:4](x)
        for i in range(0, 4):
            x_b = self.model_b[4+i](x_b)
            block_num_to_exit_in_b[i] = x_b     

        ##### mutiple modules  
        each_block_out = self.multi_exit(block_num_to_exit_in_i, block_num_to_exit_in_b, y, use_mix = use_mix)

        final_out = {'dm_conflict_out': 0 }
        out_logit_names = ['dm_conflict_out']      
        for out_logit_name in out_logit_names:
            exit_0 = f"E=0, {out_logit_name}"
            exit_1 = f"E=1, {out_logit_name}"
            exit_2 = f"E=2, {out_logit_name}"
            exit_3 = f"E=3, {out_logit_name}"
            
            ##### gating network  
            gate_in = torch.cat((each_block_out[exit_0],each_block_out[exit_1],each_block_out[exit_2],each_block_out[exit_3]),1)
            x_gate = self.gate_fc(gate_in)
            pr_gate = F.softmax(x_gate, dim=1)         
            logits_gate_i = torch.stack([each_block_out[exit_0].detach(), each_block_out[exit_1].detach(), each_block_out[exit_2].detach(), each_block_out[exit_3].detach()], dim=-1)
            logits_gate_i = logits_gate_i * pr_gate.view(pr_gate.size(0), 1, pr_gate.size(1))
            logits_gate_i = logits_gate_i.sum(-1)
            final_out[out_logit_name] = logits_gate_i            
             
        return each_block_out,final_out
