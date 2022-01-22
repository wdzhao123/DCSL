import torch
import network.pytorch_ssim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
class wmyloss_cos(nn.Module):
    def __init__(self):
        super(wmyloss_cos, self).__init__()
        self.flag=-1
        self.class_num = 20
        # self.beta = 100
        # self.discretization = "UD"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, pred, gt,l_learner,pred_class):
        gt_person_sum=[]
        for i in range(len(gt)):
            gt_person_sum.append(gt[i].sum()/100)#除以100是因为前面乘以了100,gt_person_sum是图像的人数之和
        label_class=self.gen_label(gt_person_sum)
        kl_loss_1 = nn.KLDivLoss().cuda()
        pred_class=pred_class.double()
        loss_cls = kl_loss_1(torch.log(pred_class), label_class)  # 门控分类的损失项
        loss_cls=loss_cls.float()
        loss_fn = nn.MSELoss()
        loss1 = loss_fn(pred, gt)
        loss = loss1+ loss_cls*1
        return loss

    def gen_label(self,gt_person_sum): #返回的size为4*6（4是Batchsize，6是分类向量长度，即类别个数）
        length=len(gt_person_sum)
        sigma=30.0
        list_gt=[]
        for i in range(length):
            gt_sum_i=gt_person_sum[i]
            mean=gt_sum_i.cpu().detach().numpy()
            # axis_x=np.linspace(0,13000,13000)#高斯分布横轴
            # axis_x = np.linspace(0, 600, 600)  # 高斯分布横轴
            axis_x = np.linspace(0, 3200, 3200)
            y_num_gaussion=self.normal_distribution(axis_x,mean,sigma)#纵轴-人数，生成基于人数的高斯分布
            # y_num_gaussion/=y_num_gaussion.sum()

            # plt.plot(axis_x,y_num_gaussion, 'r', label='m=mean,sig=10')
            # plt.legend()
            # plt.grid()
            # plt.show()
            # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:200]),
            #                            np.sum(y_num_gaussion[200:400]),
            #                            np.sum(y_num_gaussion[400:600])
            #                            )).transpose()
            # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:100]),
            #                            np.sum(y_num_gaussion[100:200]),
            #                            np.sum(y_num_gaussion[200:300]),
            #                            np.sum(y_num_gaussion[300:400]),
            #                            np.sum(y_num_gaussion[400:500]),
            #                            np.sum(y_num_gaussion[500:600])
            #                            )).transpose()
            # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:50]),
            #                            np.sum(y_num_gaussion[50:100]),
            #                            np.sum(y_num_gaussion[100:150]),
            #                            np.sum(y_num_gaussion[150:200]),
            #                            np.sum(y_num_gaussion[200:250]),
            #                            np.sum(y_num_gaussion[250:300]),
            #                            np.sum(y_num_gaussion[300:350]),
            #                            np.sum(y_num_gaussion[350:400]),
            #                            np.sum(y_num_gaussion[400:450]),
            #                            np.sum(y_num_gaussion[450:500]),
            #                            np.sum(y_num_gaussion[500:550]),
            #                            np.sum(y_num_gaussion[550:600])
            #                            )).transpose()
            y_level_gauss=np.vstack((np.sum(y_num_gaussion[0:300]),
                                     np.sum(y_num_gaussion[300:600]),
                                     np.sum(y_num_gaussion[600:900]),
                                     np.sum(y_num_gaussion[900:1200]),
                                     np.sum(y_num_gaussion[1200:1500]),
                                     np.sum(y_num_gaussion[1500:1800]),
                                     np.sum(y_num_gaussion[1800:2100]),
                                     np.sum(y_num_gaussion[2100:2400]),
                                     np.sum(y_num_gaussion[2400:2700]),
                                     np.sum(y_num_gaussion[2700:3000]),
                                     np.sum(y_num_gaussion[3000:3200])
                                     )).transpose()
            # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:270]),
            #                            np.sum(y_num_gaussion[270:540]),
            #                            np.sum(y_num_gaussion[540:810]),
            #                            np.sum(y_num_gaussion[810:1080]),
            #                            np.sum(y_num_gaussion[1080:1350]),
            #                            np.sum(y_num_gaussion[1350:1620]),
            #                            np.sum(y_num_gaussion[1620:1890]),
            #                            np.sum(y_num_gaussion[1890:2160]),
            #                            np.sum(y_num_gaussion[2160:2430]),
            #                            np.sum(y_num_gaussion[2430:2700]),
            #                            np.sum(y_num_gaussion[2700:2970]),
            #                            np.sum(y_num_gaussion[2970:3200])
            #                            )).transpose()
            # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:2]),
            #                            np.sum(y_num_gaussion[2:4]),
            #                            np.sum(y_num_gaussion[4:8]),
            #                            np.sum(y_num_gaussion[8:15]),
            #                            np.sum(y_num_gaussion[15:30]),
            #                            np.sum(y_num_gaussion[30:57]),
            #                            np.sum(y_num_gaussion[57:112]),
            #                            np.sum(y_num_gaussion[112:221]),
            #                            np.sum(y_num_gaussion[221:434]),
            #                            np.sum(y_num_gaussion[434:851]),
            #                            np.sum(y_num_gaussion[851:1671]),
            #                            np.sum(y_num_gaussion[1671:3200])
            #                            )).transpose()
            # # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:100]),
            # #                            np.sum(y_num_gaussion[100:200]),
            # #                            np.sum(y_num_gaussion[200:300]),
            # #                            np.sum(y_num_gaussion[300:400]),
            # #                            np.sum(y_num_gaussion[400:500]),
            # #                            np.sum(y_num_gaussion[500:600]),
            # #                            np.sum(y_num_gaussion[600:900]),
            # #                            np.sum(y_num_gaussion[900:1200]),
            # #                            np.sum(y_num_gaussion[1200:1500]),
            # #                            np.sum(y_num_gaussion[1500:1800]),
            # #                            np.sum(y_num_gaussion[1800:2100]),
            # #                            np.sum(y_num_gaussion[2100:2400]),
            # #                            np.sum(y_num_gaussion[2400:2700]),
            # #                            np.sum(y_num_gaussion[2700:3000]),
            # #                            np.sum(y_num_gaussion[3000:5000])
            # #                            )).transpose()
            # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:50]),
            #                            np.sum(y_num_gaussion[50:100]),
            #                            np.sum(y_num_gaussion[100:150]),
            #                            np.sum(y_num_gaussion[150:200]),
            #                            np.sum(y_num_gaussion[200:250]),
            #                            np.sum(y_num_gaussion[250:300]),
            #                            np.sum(y_num_gaussion[300:350]),
            #                            np.sum(y_num_gaussion[350:400]),
            #                            np.sum(y_num_gaussion[400:450]),
            #                            np.sum(y_num_gaussion[450:500]),
            #                            np.sum(y_num_gaussion[500:550]),
            #                            np.sum(y_num_gaussion[550:600])
            #                            )).transpose()
            # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:1000]),
            #                            np.sum(y_num_gaussion[1000:2000]),
            #                            np.sum(y_num_gaussion[2000:3000]),
            #                            np.sum(y_num_gaussion[3000:4000]),
            #                            np.sum(y_num_gaussion[4000:5000]),
            #                            np.sum(y_num_gaussion[5000:6000]),
            #                            np.sum(y_num_gaussion[6000:7000]),
            #                            np.sum(y_num_gaussion[7000:8000]),
            #                            np.sum(y_num_gaussion[8000:9000]),
            #                            np.sum(y_num_gaussion[9000:10000]),
            #                            np.sum(y_num_gaussion[10000:11000]),
            #                            np.sum(y_num_gaussion[11000:12000]),
            #                            np.sum(y_num_gaussion[12000:13000])
            #                            )).transpose()
            # y_level_gauss = np.vstack((np.sum(y_num_gaussion[0:2]),
            #                            np.sum(y_num_gaussion[2:5]),
            #                            np.sum(y_num_gaussion[5:11]),
            #                            np.sum(y_num_gaussion[11:25]),
            #                            np.sum(y_num_gaussion[25:56]),
            #                            np.sum(y_num_gaussion[56:126]),
            #                            np.sum(y_num_gaussion[126:282]),
            #                            np.sum(y_num_gaussion[282:631]),
            #                            np.sum(y_num_gaussion[631:1413]),
            #                            np.sum(y_num_gaussion[1413:3162]),
            #                            np.sum(y_num_gaussion[3162:7079]),
            #                            np.sum(y_num_gaussion[7079:13000])
            #                            )).transpose()
            # y_level_gauss=
            y_level_gauss=torch.from_numpy(y_level_gauss).cuda().double()
            list_gt.append(y_level_gauss)
        class_gt=list_gt[0]
        for i in range(1,length):
            class_gt=torch.cat((class_gt,list_gt[i]))
        # class_gt=torch.from_numpy(class_gt).cuda().float()
        return class_gt

    def normal_distribution(self,x, mean, sigma):
        return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

    def _create_ord_label(self, gt):
        N, H, W = gt.shape
        # print("gt shape:", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt) / (self.beta )
        label = label.long()

        one_hot_label = np.zeros(shape=(label.shape[0], 50))  ##生成全0矩阵
        one_hot_label[np.arange(0, label.shape[0]), label] = 1  ##相应标签位置置1
        print(one_hot_label)

        # mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
        #     .view(1, self.ord_num, 1, 1).to(gt.device)
        # mask = mask.repeat(N, 1, H, W).contiguous().long()
        # mask = (mask > label)
        # ord_c0[mask] = 0
        # ord_c1 = 1 - ord_c0
        # implementation according to the paper.
        # ord_label = torch.ones(N, self.ord_num * 2, H, W).to(gt.device)
        # ord_label[:, 0::2, :, :] = ord_c0
        # ord_label[:, 1::2, :, :] = ord_c1
        # reimplementation for fast speed.
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def get_one_hot(self,label, N):
        size = list(label.size())
        label = label.view(-1).cpu()  # reshape 为向量
        label=label.long()
        # label=label.cpu().detach().numpy()
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
        size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
        return ones.view(*size)
