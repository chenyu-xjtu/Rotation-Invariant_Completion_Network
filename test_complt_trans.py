from tqdm import tqdm
from utils.train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from dataset_test import MVP
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from utils.vis_utils import plot_single_pcd
from utils.train_utils import *
import torch.optim as optim
import torch

def train():
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    dataset = MVP(prefix="train", npoints=args.num_points)
    dataset_test = MVP(prefix="test", npoints=args.num_points)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)

    cascade_gan = (args.model_name == 'cascade')
    net_d = None
    if cascade_gan:
        net_d = torch.nn.DataParallel(model_module.Discriminator(args))
        net_d.cuda()
        net_d.module.apply(model_module.weights_init)

    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        if cascade_gan:
            net_d.module.load_state_dict(ckpt['D_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    val(net, dataloader_test)


def val(net, dataloader_test):
    logging.info('Testing...')

    net.module.eval()

    idx_to_plot = [i for i in range(0, 41600, 100)]

    if args.save_vis:
        save_gt_path = os.path.join(log_dir, 'pics_trans', 'gt')
        save_partial_path = os.path.join(log_dir, 'pics_trans', 'partial')
        save_completion_path = os.path.join(log_dir, 'pics_trans', 'completion')
        os.makedirs(save_gt_path, exist_ok=True)
        os.makedirs(save_partial_path, exist_ok=True)
        os.makedirs(save_completion_path, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader_test)):
            label, incomplete_pointcloud1, incomplete_pointcloud2, complete_pointcloud1, complete_pointcloud2, \
                rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = data
            # mean_feature = None
            inputs = incomplete_pointcloud2.float().cuda().contiguous()
            gt = complete_pointcloud2.float().cuda().contiguous()

            # result_dict = net(inputs, gt, is_training=False, mean_feature=mean_feature)
            result_dict = net(inputs, gt, is_training=False)

            #test
            if args.save_vis:
                for z in range(args.batch_size):
                    idx = i * args.batch_size + z
                    if idx in idx_to_plot:
                        pic = 'object_%d.png' % idx
                        plot_single_pcd(result_dict['out2'][z].cpu().numpy(), os.path.join(save_completion_path, pic))
                        plot_single_pcd(inputs[:, :, :3][z].cpu().numpy(), os.path.join(save_partial_path, pic))
                        plot_single_pcd(gt[z].cpu().numpy(), os.path.join(save_gt_path, pic))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])
    train()



