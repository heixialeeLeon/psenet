# -*- coding: utf-8 -*-
import cv2
import os
import argparse
from configs import config_tips as config

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

import shutil
import glob
import time
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torch.utils.data as Data
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

#from dataset.data_utils import MyDataset
from dataset.dataset_tips import MyDataset
from models import PSENet
from models.loss import PSELoss
from utils.utils import load_checkpoint, save_checkpoint, setup_logger
from pse import decode as pse_decode
from cal_recall import cal_recall_precison_f1

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# training parameters
# parser.add_argument("--image_size", type=int, default=256,help="training patch size")
parser.add_argument("--train_dir", type=str, default=config.train_folder,help="train data dir location")
parser.add_argument("--test_dir", type=str, default=config.test_folder,help="test data dir location")
parser.add_argument("--batch_size", type=int, default=config.batch_size,help="batch size")
parser.add_argument("--epochs", type=int, default=config.epochs,help="number of epochs")
parser.add_argument("--save_per_epoch", type=int, default=config.save_per_epoch,help="number of epochs")
parser.add_argument("--lr", type=float, default=config.lr, help="learning rate")
parser.add_argument("--steps_show", type=int, default=100,help="steps per epoch")
parser.add_argument("--scheduler_step", type=int, default=10,help="scheduler_step for epoch")
parser.add_argument("--weight", type=str, default=None,help="weight file for restart")
parser.add_argument("--output_path", type=str, default="checkpoint",help="checkpoint dir")
parser.add_argument("--devices", type=str, default="cuda",help="device description")
parser.add_argument("--image_channels", type=int, default=3,help="batch image_channels")
parser.add_argument("--resume_model", type=str, default=config.resume_model, help="resume model path")

args = parser.parse_args()
print(args)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# learning rate的warming up操作
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

# learning rate的warming up操作
def adjust_learning_rate_leon(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch // 50))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_epoch(net, optimizer, scheduler, train_loader, device, criterion, epoch, all_step, writer, logger):
    net.train()
    train_loss = 0.
    start = time.time()
    #scheduler.step()
    #lr = adjust_learning_rate_leon(optimizer, epoch)
    lr = scheduler.get_lr()[0]
    for i, (images, labels, training_mask) in enumerate(train_loader):
        cur_batch = images.size()[0]
        images, labels, training_mask = images.to(device), labels.to(device), training_mask.to(device)
        # Forward
        y1 = net(images)
        loss_c, loss_s, loss = criterion(y1, labels, training_mask)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loss_c = loss_c.item()
        loss_s = loss_s.item()
        loss = loss.item()
        cur_step = epoch * all_step + i
        writer.add_scalar(tag='Train/loss_c', scalar_value=loss_c, global_step=cur_step)
        writer.add_scalar(tag='Train/loss_s', scalar_value=loss_s, global_step=cur_step)
        writer.add_scalar(tag='Train/loss', scalar_value=loss, global_step=cur_step)
        writer.add_scalar(tag='Train/lr', scalar_value=lr, global_step=cur_step)

        if i % config.display_interval == 0:
            batch_time = time.time() - start
            logger.info(
                '[{}/{}], [{}/{}], step: {}, {:.3f} samples/sec, batch_loss: {:.4f}, batch_loss_c: {:.4f}, batch_loss_s: {:.4f}, time:{:.4f}, lr:{}'.format(
                    epoch, args.epochs, i, all_step, cur_step, config.display_interval * cur_batch / batch_time,
                    loss, loss_c, loss_s, batch_time, lr))
            start = time.time()

        if i % config.show_images_interval == 0:
            if config.display_input_images:
                # show images on tensorboard
                x = vutils.make_grid(images.detach().cpu(), nrow=4, normalize=True, scale_each=True, padding=20)
                writer.add_image(tag='input/image', img_tensor=x, global_step=cur_step)

                show_label = labels.detach().cpu()
                b, c, h, w = show_label.size()
                show_label = show_label.reshape(b * c, h, w)
                show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=config.n, normalize=False, padding=20,
                                              pad_value=1)
                writer.add_image(tag='input/label', img_tensor=show_label, global_step=cur_step)

            if config.display_output_images:
                y1 = torch.sigmoid(y1)
                show_y = y1.detach().cpu()
                b, c, h, w = show_y.size()
                show_y = show_y.reshape(b * c, h, w)
                show_y = vutils.make_grid(show_y.unsqueeze(1), nrow=config.n, normalize=False, padding=20, pad_value=1)
                writer.add_image(tag='output/preds', img_tensor=show_y, global_step=cur_step)
    writer.add_scalar(tag='Train_epoch/loss', scalar_value=train_loss / all_step, global_step=epoch)
    scheduler.step()
    return train_loss / all_step, lr


def eval(model, save_path, test_path, device):
    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    img_path = os.path.join(test_path, 'img')
    gt_path = os.path.join(test_path, 'gt')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    long_size = 2240
    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    for img_path in tqdm(img_paths, desc='test models'):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        #if max(h, w) > long_size:
        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            preds, boxes_list = pse_decode(preds[0], config.scale)
            scale = (preds.shape[1] * 1.0 / w, preds.shape[0] * 1.0 / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    # 开始计算 recall precision f1
    result_dict = cal_recall_precison_f1(gt_path, save_path)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']

def save_model(model,epoch):
    '''save model for eval'''
    ckpt_name = '/pse_epoch_{}.pth'.format(epoch)
    path = "checkpoint"
    if not os.path.exists(path):
        os.mkdir(path)
    path_final = path + ckpt_name
    print('Saving checkpoint to: {}\n'.format(path_final))
    torch.save(model.state_dict(), path_final)

def resume_model(model, model_path):
    print("Resume model from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))

def main():
    if config.output_dir is None:
        config.output_dir = 'output'
    if config.restart_training:
        shutil.rmtree(config.output_dir, ignore_errors=True)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    logger = setup_logger(os.path.join(config.output_dir, 'train_log'))
    logger.info(config.print())

    torch.manual_seed(config.seed)  # 为CPU设置随机种子
    if config.gpu_id is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info('train with gpu {} and pytorch {}'.format(config.gpu_id, torch.__version__))
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(config.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(config.seed)  # 为所有GPU设置随机种子
    else:
        logger.info('train with cpu and pytorch {}'.format(torch.__version__))
        device = torch.device("cpu")

    train_data = MyDataset(args.train_dir, data_shape=config.data_shape, n=config.n, m=config.m,
                           transform=transforms.ToTensor())
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                   num_workers=int(config.workers))

    writer = SummaryWriter(config.output_dir)
    model = PSENet(backbone=config.backbone, pretrained=config.pretrained, result_num=config.n, scale=config.scale)
    if not config.pretrained and not config.restart_training:
        model.apply(weights_init)

    if args.resume_model:
        resume_model(model,args.resume_model)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    # dummy_input = torch.autograd.Variable(torch.Tensor(1, 3, 600, 800).to(device))
    # writer.add_graph(models=models, input_to_model=dummy_input)
    criterion = PSELoss(Lambda=config.Lambda, ratio=config.OHEM_ratio, reduction='mean')
    # optimizer = torch.optim.SGD(models.parameters(), lr=config.lr, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if config.checkpoint != '' and not config.restart_training:
        start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
        start_epoch += 1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                         last_epoch=start_epoch)
    else:
        start_epoch = config.start_epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

    all_step = len(train_loader)
    logger.info('train dataset has {} samples,{} in dataloader'.format(train_data.__len__(), all_step))
    epoch = 0
    best_model = {'recall': 0, 'precision': 0, 'f1': 0, 'models': ''}
    try:
        for epoch in range(start_epoch, args.epochs):
            start = time.time()
            train_loss, lr = train_epoch(model, optimizer, scheduler, train_loader, device, criterion, epoch, all_step,
                                         writer, logger)
            logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
                epoch, config.epochs, train_loss, time.time() - start, lr))

            if epoch % args.save_per_epoch == 0:
                save_model(model, epoch)
        writer.close()

    except KeyboardInterrupt:
        save_checkpoint('{}/final.pth'.format(config.output_dir), model, optimizer, epoch, logger)
    finally:
        if best_model['models']:
            logger.info(best_model)


if __name__ == '__main__':
    main()
