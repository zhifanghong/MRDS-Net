import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
#parser.add_argument('--root_path', type=str, default='../data/2020COVIDSeg_TrainingSet/', help='Name of Experiment')
parser.add_argument('--root_path', type=str, default='../data/LITS/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='ds_vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/UAMT_unlabel/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../LITS_test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "h5/" +item.replace('\n', '')+"_h5.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'LITSl_model_iter_' + str(epoch_num) + 'lr0.001.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(256, 256, 16), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(30000)
    print(metric)
