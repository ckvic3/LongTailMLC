import torch.nn as nn
import torch
from copy import deepcopy
from torchvision.models.resnet import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from .pfc import PFC
from .head import ClsHead

class GroupModel(nn.Module):
    def __init__(self,label_groups,num_classes=80,mode="global",freeze_max_layer=0) -> None:
        """
        
        mode: [ local, global, fusion ] 
        """
        super(GroupModel,self).__init__()
        self.label_groups = label_groups
        self.num_classes = num_classes
        self.mode = mode

        self.backbone = IntermediateLayerGetter(resnet50(pretrained=True),return_layers={"layer3":"layer3"})
        
        self._freeze_layers(freeze_max_layer)

        model = resnet50(pretrained=True)
        self.group_heads = nn.ModuleList()

        if self.mode == "local" or self.mode == "fusion":
            for label_group in label_groups:
                self.group_heads.append(nn.Sequential(
                    deepcopy(model.layer4),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(start_dim=1),
                    PFC(in_channels=2048,out_channels=256,dropout=0.5),
                    ClsHead(in_channels=256,num_classes=len(label_group))
                ))

        if self.mode == "global" or self.mode == "fusion":
            self.group_heads.append(nn.Sequential(
                    deepcopy(model.layer4),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(start_dim=1),
                    PFC(in_channels=2048,out_channels=256,dropout=0.5),
                    ClsHead(in_channels=256,num_classes=num_classes)
                ))

    def _freeze_layers(self, freeze_max_layer):
        if freeze_max_layer <= 0:
            return
        
        # 冻结第一层之前的module
        for name, module in self.backbone.named_modules():
            if "layer" not in name:
                for _,param in module.named_parameters():
                    param.requires_grad = False

        # 冻结小于freeze_max_layer的各层module
        for layer in range(1,freeze_max_layer+1):
            for _,param in self.backbone["layer"+str(layer)].named_parameters():
                param.requires_grad = False

    def forward(self,x):
        device = (torch.device('cuda')
                  if x.is_cuda
                  else torch.device('cpu'))
        b = x.shape[0]
        x = self.backbone(x)["layer3"]
        
        global_logit = local_logit = None

        if self.mode == "global" or self.mode == "fusion":
            global_logit = self.group_heads[-1](x)
        
        if self.mode == "local" or self.mode == "fusion":
            local_logit = torch.zeros(size=[b,self.num_classes]).to(device)
            for i, label_group in enumerate(self.label_groups):
                local_logit[:, label_group] = self.group_heads[i](x)
        
        if self.training:
            return (local_logit, global_logit)
        else:
            return (local_logit + global_logit) / 2.0 


