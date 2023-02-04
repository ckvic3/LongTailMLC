from .pfc import PFC
from .head import ClsHead
from .groupModel import GroupModel
from .baseModel import BaseModel
from mmcv import Config

def createModelFactory(cfg:Config):
    if cfg.model['name'] == "base":
        return BaseModel(num_classes=cfg.model['num_classes'])
    elif cfg.model['name'] == "group":
        return GroupModel(label_groups=cfg.model['label_groups'],num_classes=cfg.model['num_classes'], mode = cfg.model['mode'],freeze_max_layer= cfg.model['freeze_max_layer'])