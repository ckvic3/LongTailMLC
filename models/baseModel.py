from torchvision.models.resnet import resnet50
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn as nn
from models import PFC, ClsHead

class BaseModel(nn.Module):
    def __init__(self,num_classes=80) -> None:
        super(BaseModel,self).__init__()
        self.backbone = IntermediateLayerGetter(resnet50(pretrained=True),return_layers={"avgpool":"avgpool"})
        self.neck = PFC(in_channels=2048,out_channels=256,dropout=0.5)
        self.fc = ClsHead(in_channels=256, num_classes=num_classes)
        
    def forward(self,x):
        x = self.backbone(x)["avgpool"]
        x = self.neck(x)
        x = self.fc(x)
        return x



