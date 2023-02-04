import torch.nn as nn

class ClsHead(nn.Module):
    """Simplest classification head, with only one fc layer for classification"""

    def __init__(self,
                 in_channels=256,
                 num_classes=80, ):
        super(ClsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self._init_weights()

    def _init_weights(self):
        print("cls head init")
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return cls_score