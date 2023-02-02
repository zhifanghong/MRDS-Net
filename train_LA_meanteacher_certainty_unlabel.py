import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import nibabel as nib
import torch.nn as nn
#######
#######   train.list的1-149为有标签数据  150-499为无标签数据
#######
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from networks.MRVnet import MRVNet
from networks.AttUnet import U_Net
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2020COVIDSeg_TrainingSet/', help='Name of Experiment')
#parser.add_argument('--root_path', type=str, default='../data/NIH/', help='Name of Experiment')
#parser.add_argument('--root_path', type=str, default='../data/client1/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--threshold', type=float,  default='0.7', help='the stable threshold')
parser.add_argument('--isCheckPoint', type=bool,  default=False, help='checkpoint')
parser.add_argument('--CheckPointe_poch', type=int,  default=4000, help='checkpoint_epoch')
parser.add_argument('--CheckPoint_path', type=str,  default="../model/UAMT_unlabel/", help='checkpoint')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112 , 112, 48)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_stabilization_weight(epoch):
    stabilization_scale = 100
    stabilization_scale_rampup = 10000
    return stabilization_scale * ramps.sigmoid_rampup(epoch, stabilization_scale_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = MRVNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        #net = U_Net(img_ch= 1 , output_ch= num_classes)
        net = nn.DataParallel(net)
        model = net.cuda()
        # if ema:
        #     for param in model.parameters():
        #         param.detach_()
        return model

    r_model = create_model()
    l_model = create_model()


    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    #39
    labeled_idxs = list(range(149))
    unlabeled_idxs = list(range(149, 600))
    # labeled_idxs = list(range(20))
    # unlabeled_idxs = list(range(20, 39))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    r_model.train()
    l_model.train()
    # r_optimizer = optim.Adam(r_model.parameters() , lr = base_lr)
    # l_optimizer = optim.Adam(l_model.parameters() , lr = base_lr)
    r_optimizer = optim.SGD(r_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    l_optimizer = optim.SGD(l_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
        consistency_criterion_scalar = losses.softmax_mse_loss_scalar
        stabilization_criterion = losses.softmax_mse_loss_scalar
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
        stabilization_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    l_model.train()
    r_model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            l_loss = 0
            r_loss = 0
            # print('fetch data cnost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']   #读取批数据
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            r_noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            l_noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)

            r_input = volume_batch
            l_input = r_input.clone()
            rn_input = volume_batch + r_noise
            ln_input = volume_batch + l_noise

            r_outputs = r_model(r_input)
            rn_outputs = r_model(rn_input)

            l_outputs = l_model(l_input)
            ln_outputs = l_model(ln_input)


            #segmentation  loss
            focal_loss = losses.FocalLoss(alpha=0.5)
            l_celoss = F.cross_entropy(l_outputs[:labeled_bs], label_batch[:labeled_bs].long())
            #l_celoss = focal_loss(l_outputs[:labeled_bs], label_batch[:labeled_bs].long())
            l_outputs_soft = F.softmax(l_outputs, dim=1)
            l_diceloss = losses.dice_loss(l_outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

            r_celoss = F.cross_entropy(r_outputs[:labeled_bs], label_batch[:labeled_bs].long())
            #r_celoss = focal_loss(r_outputs[:labeled_bs], label_batch[:labeled_bs].long())
            r_outputs_soft = F.softmax(r_outputs, dim=1)
            r_diceloss = losses.dice_loss(r_outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)


            #consistency loss
            consistency_weight = get_current_consistency_weight(iter_num / 750)
            l_conloss = consistency_criterion_scalar(l_outputs, ln_outputs) / args.batch_size
            r_conloss = consistency_criterion_scalar(r_outputs, rn_outputs) / args.batch_size
            #ln_outputs_soft = F.softmax(ln_outputs,dim = 1)
            # rn_outputs_soft = F.softmax(rn_outputs,dim = 1)
            # l_con_outputs_soft = l_outputs_soft.clone()
            # r_con_outputs_soft = r_outputs_soft.clone()
            # l_conloss = losses.dice_loss(l_con_outputs_soft[:labeled_bs, 1, :, :, :],ln_outputs_soft[:labeled_bs, 1, :, :, :])
            # r_conloss = losses.dice_loss(r_con_outputs_soft[:labeled_bs, 1, :, :, :],rn_outputs_soft[:labeled_bs, 1, :, :, :])


            #stabilization loss
            unlabeled_batchsize = args.batch_size - args.labeled_bs
            for unlabeled_index in range(args.labeled_bs, args.batch_size):
                l_sample_ce = consistency_criterion_scalar(l_outputs[unlabeled_index].detach(),ln_outputs[unlabeled_index].detach())
                r_sample_ce = consistency_criterion_scalar(r_outputs[unlabeled_index].detach(),rn_outputs[unlabeled_index].detach())
                l_stable = False
                if((1 - l_sample_ce) > args.threshold ):
                    l_stable = True

                r_stable = False
                if((1 - r_sample_ce) > args.threshold ):
                    r_stable = True

                l_target_outputs = r_outputs.clone()
                r_target_outputs = l_outputs.clone()
                stabilization_weight = get_current_stabilization_weight(iter_num/3)
                if (l_stable and r_stable):

                    if(l_sample_ce < r_sample_ce):
                        #l->r
                        l_target_outputs[unlabeled_index] = l_outputs[unlabeled_index].detach()

                    else:
                        #r->l
                        r_target_outputs[unlabeled_index] = r_outputs[unlabeled_index].detach()
            r_staloss = stabilization_criterion(r_outputs, l_target_outputs)
            #r_loss = r_loss + (stabilization_weight * r_staloss) / unlabeled_batchsize
            l_staloss = stabilization_criterion(l_outputs, r_target_outputs)
            #l_loss = l_loss + (stabilization_weight * l_staloss) / unlabeled_batchsize

            l_loss =(( 0.3*l_celoss + 0.7 *l_diceloss)) + (consistency_weight * l_conloss) + (stabilization_weight * l_staloss) / unlabeled_batchsize
            r_loss =((0.3*r_celoss + 0.7 *r_diceloss)) + (consistency_weight * r_conloss) + (stabilization_weight * r_staloss) / unlabeled_batchsize
            # update model
            l_optimizer.zero_grad()
            l_loss.backward()
            l_optimizer.step()

            r_optimizer.zero_grad()
            r_loss.backward()
            r_optimizer.step()


            if iter_num % 100 == 0:
                print('/n' + "iter_num:" + str(iter_num) + "   r_loss:" + str(r_loss.item())+ "    l_loss" + str(l_loss.item()))
                print('/n' + "iter_num:" + str(iter_num) + "   r_celoss:" + str(r_celoss.item())+ "    l_celoss" + str(l_celoss.item()))
                print("l_diceloss:" + str(l_diceloss.item()) + "r_diceloss:" + str(r_diceloss.item()))
            iter_num = iter_num + 1

            writer.add_scalar("r_loss/r_loss", r_loss, iter_num)
            writer.add_scalar("l_loss/l_loss", l_loss, iter_num)
            writer.add_scalar("r_loss/r_diceloss", r_diceloss, iter_num)
            writer.add_scalar("l_loss/l_diceloss", l_diceloss, iter_num)
            writer.add_scalar("r_loss/r_celoss", r_celoss, iter_num)
            writer.add_scalar("l_loss/l_celoss", l_celoss, iter_num)
            writer.add_scalar("r_loss/r_conloss", r_conloss, iter_num)
            writer.add_scalar("l_loss/l_conloss", l_conloss, iter_num)
            writer.add_scalar("r_loss/r_staloss", r_staloss, iter_num)
            writer.add_scalar("l_loss/l_staloss", l_staloss, iter_num)
            writer.add_scalar("weight/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("weight/stabilization_weight", stabilization_weight, iter_num)

            if iter_num % 100 == 0:
                image = volume_batch[0, 0:1, :, :, 10:51:5].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(l_outputs_soft[0, :, :, :, 10:51:5], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 10:51:5].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)
            # change lr
            if iter_num % 5000 == 0:
                lr_ = base_lr * 0.2 ** (iter_num // 5000)
                for param_group in r_optimizer.param_groups:
                    param_group['lr'] = lr_
                for param_group in l_optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                r_save_mode_path = os.path.join(snapshot_path, 'covidr_model_iter_' + str(iter_num) + '.pth')
                l_save_mode_path = os.path.join(snapshot_path, 'covidl_model_iter_' + str(iter_num) + '.pth')
                torch.save(r_model.state_dict(), r_save_mode_path)
                torch.save(l_model.state_dict(), l_save_mode_path)
                logging.info("save model to {}".format(r_save_mode_path))
                logging.info("save model to {}".format(l_save_mode_path))

        if iter_num >= max_iterations:
            break
        time1 = time.time()
        if iter_num >= max_iterations:
            break
    r_model_save_mode_path = os.path.join(snapshot_path, '256covidr_model_iter_'+str(max_iterations)+'lr'+str(base_lr)+'.pth')
    l_model_save_mode_path = os.path.join(snapshot_path, '256covidl_model_iter_'+str(max_iterations)+'lr'+str(base_lr)+'.pth')
    torch.save(r_model.state_dict(), r_model_save_mode_path)
    torch.save(l_model.state_dict(), l_model_save_mode_path)
    logging.info("save model to {}".format(snapshot_path))
    writer.close()
