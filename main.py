from models import createModelFactory
import torch
from dataset import getDataloader
from mmcv import Config
import argparse
from losses import createLossFuntion
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str,default="/home/pengpeng/LongTailMLC/configs/coco/coco_base.py")
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--lr', default=1e-4, type=float)
args = parser.parse_args()

if __name__ == '__main__':
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(dict(args._get_kwargs()))
    model = createModelFactory(cfg).cuda()
    trainLoader, valLoader = getDataloader(cfg)
    criterion = createLossFuntion(cfg)
    
    optimizer = torch.optim.SGD(params=model.parameters(),weight_decay=1e-4,momentum=0.9,lr=cfg.lr)
    
    ema_loss = AverageMeter()
    now = datetime.datetime.now()
    output_path = f"{now.date()}-{now.hour}-{now.minute}".replace(' ', '-')
    output_path = os.path.join(cfg.output_path,output_path)
    os.makedirs(output_path)
    writer = SummaryWriter(log_dir=output_path)

    cur_step = -1
    for epoch in range(cfg.epochs):
        for i,(imgs,targets,masks) in enumerate(trainLoader):
            cur_step += 1
            imgs = imgs.cuda()
            targets = targets.cuda()
            masks = masks.cuda()
            
            output = model(imgs)
            if cfg.model['name'] == 'base':
                loss = criterion(output.float(), targets, masks)
            else:
                if cfg.model['mode'] == 'local':
                    loss = criterion(output[0].float(), targets, masks)
                elif cfg.model['mode'] == 'global':
                    loss = criterion(output[1].float(), targets, masks)
                elif cfg.model['mode'] == 'fusion':
                    loss = criterion(output[0].float(), targets, masks) + criterion(output[1].float(), targets, masks)
                else:
                    raise NotImplementedError
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ema update loss
            ema_loss.update(loss.detach().cpu().numpy(),n=imgs.shape[0])

            writer.add_scalar('train/ema_loss',ema_loss.ema,cur_step)
            writer.add_scalar('train/loss',loss.detach().cpu().item(), cur_step)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], cur_step)
