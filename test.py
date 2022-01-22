import torchvision.transforms as standard_transforms
import datasets.data_preprocess as img_preprocess
from models.crowd_model import CrowdCounter
from matplotlib import pyplot as plt
from torch.autograd import Variable
from PIL import Image, ImageOps
from tools.num_cal import *
from tqdm import tqdm
import scipy.io as sio
from absl import flags
import pandas as pd
import numpy as np
import torch
import time
import sys
import os
import cv2

# -------------parameters setting-----------
message = '在test集上测试,输出最终結果'
flags.DEFINE_list('gpu_id', [0], 'the GPU_ID')
flags.DEFINE_string('data_mode', 'SHA', 'the dataset you run your model')
flags.DEFINE_string('net', 'model', 'Number of train steps.')
# flags.DEFINE_string('net', 'SFAoatt', 'Number of train steps.')
flags.DEFINE_string('model_path',
                    "weights/QNRF.pth",
                    'model path.')
# 改此处决定测试训练集还是测试集
flags.DEFINE_string('data_dir', "test", 'data_dir.')
flags.DEFINE_bool('if_de', False, 'if de dataset.')
# 真值放大
gt_log_para = 100.
FLAGS = flags.FLAGS
FLAGS(sys.argv)

torch.cuda.set_device(0)
# torch.backends.cudnn.benchmark = True
flags.mark_flag_as_required('gpu_id')
flags.mark_flag_as_required('data_mode')
flags.mark_flag_as_required('net')
flags.mark_flag_as_required('model_path')
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('if_de')
data_mode = FLAGS.data_mode
data_dir = FLAGS.data_dir
data_path = FLAGS.model_path
test_model = FLAGS.net
if_de = FLAGS.if_de

exp_name = os.path.join('test_results', test_model + '_' + data_mode,
                        time.strftime("%m-%d_%H-%M", time.localtime()))
if not os.path.exists('test_results'):
    os.mkdir('test_results')
if not os.path.exists(os.path.join('test_results', test_model + '_' + data_mode)):
    os.mkdir(os.path.join('test_results', test_model + '_' + data_mode))
if not os.path.exists(exp_name):
    os.mkdir(exp_name)
if not os.path.exists(exp_name + '/pred'):
    os.mkdir(exp_name + '/pred')
if not os.path.exists(exp_name + '/gt'):
    os.mkdir(exp_name + '/gt')
model_path = FLAGS.model_path

if data_mode == 'SHB':
    from datasets.SHB.base_cfg import cfg_data
if data_mode == 'SHA':
    from datasets.SHA.base_cfg import cfg_data
if data_mode == 'UCF50':
    from datasets.UCF50.base_cfg import cfg_data
if data_mode == 'QNRF':
    from datasets.QNRF.base_cfg import cfg_data
mean_std = cfg_data.MEAN_STD
dataRoot = os.path.join('datasets', data_mode, data_dir)

img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
restore = standard_transforms.Compose([
    img_preprocess.DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])
pil_to_tensor = standard_transforms.ToTensor()

def test(file_list, path):
    net = CrowdCounter(FLAGS.gpu_id, FLAGS.net)
    net.load_state_dict(torch.load(path, map_location='cuda:0'))
    net.cuda()
    net.eval()
    all_pics_cnt = 0
    maes = AverageMeter()
    mses = AverageMeter()
    test_logfile = exp_name + '/loger.txt'

    for filename in tqdm(file_list):
        all_pics_cnt += 1
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]
        denname = dataRoot + '/den/' + filename_no_ext + '.csv'
        den = pd.read_csv(denname, sep=',', header=None).values
        den = den.astype(np.float32, copy=False)
        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')

        img = img_transform(img)
        gt = np.sum(den)
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map,list_learner,pred_class= net.test_forward(img)
            pred_map = torch.mean(pred_map,dim=-3)

        pred_map = pred_map.cpu().data.numpy().squeeze()

        pred = np.sum(pred_map) / float(cfg_data.log_para)
        pred_map  = pred_map / np.max(pred_map + 1e-20)
        den = den / np.max(den + 1e-20)
        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False)
        den_frame.spines['bottom'].set_visible(False)
        den_frame.spines['left'].set_visible(False)
        den_frame.spines['right'].set_visible(False)
        if not os.path.exists(exp_name + '/gt_imgs'):
            os.mkdir(exp_name + '/gt_imgs')
        plt.savefig(exp_name + '/gt_imgs' + '/' + filename_no_ext + '_gt_' + str(int(gt)) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        if not os.path.exists(exp_name + '/pred_imgs'):
            os.mkdir(exp_name + '/pred_imgs')
        plt.savefig(exp_name + '/pred_imgs' + '/' + filename_no_ext + '_pred_' + str(float(pred)) + '_gt_' + str(float(gt)) + '.png', \
                    bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        maes.update(abs(gt - pred))
        mses.update((pred - gt) * (pred - gt))

    mae = maes.avg
    mse = np.sqrt(mses.avg)
    print('################################')
    print("MAE:", mae)
    print("MSE:", mse)
    with open(test_logfile, 'w') as file_out:
        file_out.write(message + str(mae) + '/' + str(mse) + '||||' + data_path)

def main():

    file_list = [filename for root, dirs, filename in os.walk(dataRoot + '/img/')]
    print('=================================================')
    print('----------net:', test_model)
    print('----------dataset:', data_mode)
    print('model_path:', model_path)
    test(file_list[0], model_path)


if __name__ == '__main__':
    main()
