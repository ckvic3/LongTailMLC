from .asl import AsymmetricLossOptimized,FocalLoss
from .bce import BCEWithLogitLoss

def createLossFuntion(cfg):
    name = cfg.loss['name']
    print("using {} loss function, param is {}".format(name,cfg.loss['param']))
    if name == 'asl':
        return AsymmetricLossOptimized(**cfg.loss['param'])
    elif name == 'focal':
        return FocalLoss(**cfg.loss['param'])
    elif name == 'bce':
        return BCEWithLogitLoss()