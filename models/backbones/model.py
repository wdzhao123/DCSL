import torch
from torch import nn
from torch.utils import model_zoo
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F

num_class_SHA=11
num_class_SHB=12
num_class_QNRF=13
num_class=num_class_QNRF

# SFA without attention
class SFAoatt(nn.Module):
    def __init__(self):
        super(SFAoatt, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.middle = Middle()
        for p in self.parameters():
            p.requires_grad=False
        self.vgg1 = VGG1()
        self.load_vgg1()
        self.gater=Gater(num_class)
        self.choice=Choice()
        self.dmp = BackEnd()
        self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)
    def forward(self, input):
        input1=input
        input = self.vgg(input)
        input1=self.vgg1(input1)
        out_middle,l_learner,l_mask = self.middle(*input)
        g,pred_class = self.gater(l_learner, input1)
        out = self.choice(l_learner, g)
        dmp_out_32 = self.dmp(out,*input)
        dmp_out_1 = self.conv_out(dmp_out_32)
        return dmp_out_1,l_mask,pred_class
    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']
        self.vgg.load_state_dict(new_dict)
    def load_vgg1(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']
        self.vgg1.load_state_dict(new_dict)
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)
        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)
        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)
        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)

        return conv2_2, conv3_3, conv4_3, conv5_3
class VGG1(nn.Module):
    def __init__(self):
        super(VGG1, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)
        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)
        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)
        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)
        return conv5_3

class Middle(nn.Module):
    def __init__(self):
        super(Middle, self).__init__()
        self.tanh=nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.conv5_0_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv5_0_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.IBN5_0_1 = IBN(512, ratio=0.5)
        self.IBN5_0_3 = IBN(512, ratio=0.5)

        self.conv5_1_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_2_1 = BaseConv(512, 1, 1, 1, activation=None, use_bn=False)

        self.conv5_2_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2_1 = BaseConv(512, 1, 1, 1, activation=None, use_bn=False)

        self.conv5_3_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2_1 = BaseConv(512, 1, 1, 1, activation=None, use_bn=False)

        self.conv5_4_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_4_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_4_2_1 = BaseConv(512, 1, 1, 1, activation=None, use_bn=False)

        self.conv5_5_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_5_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_5_2_1 = BaseConv(512, 1, 1, 1, activation=None, use_bn=False)

        self.conv5_6_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_6_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_6_2_1 = BaseConv(512, 1, 1, 1, activation=None, use_bn=False)

        self.conv5_7_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_7_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_7_2_1 = BaseConv(512, 1, 1, 1, activation=None, use_bn=False)

        self.conv5_8_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_8_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_8_2_1 = BaseConv(512, 1, 1, 1, activation=None, use_bn=False)


    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input
        input5_3=conv5_3
        input5_3 =self.conv5_0_1_1(input5_3)
        input5_3 = self.IBN5_0_1(input5_3)
        input5_3 = self.conv5_0_3_1(input5_3)
        input5_3 = self.IBN5_0_3(input5_3)
        residual5_3=conv5_3-input5_3
        learner51,mask51 = self.cal_learner_and_prior(residual5_3,input5_3,self.conv5_1_1_1,self.conv5_1_2_1,self.conv5_1_3_1)
        learner52,mask52 = self.cal_learner_and_prior(residual5_3,input5_3,self.conv5_2_1_1,self.conv5_2_2_1,self.conv5_2_3_1)
        learner53,mask53 = self.cal_learner_and_prior(residual5_3,input5_3,self.conv5_3_1_1,self.conv5_3_2_1,self.conv5_3_3_1)
        learner54, mask54 = self.cal_learner_and_prior(residual5_3, input5_3, self.conv5_4_1_1, self.conv5_4_2_1,
                                                       self.conv5_4_3_1)
        learner55, mask55 = self.cal_learner_and_prior(residual5_3, input5_3, self.conv5_5_1_1, self.conv5_5_2_1,
                                                       self.conv5_5_3_1)
        learner56, mask56 = self.cal_learner_and_prior(residual5_3, input5_3, self.conv5_6_1_1, self.conv5_6_2_1,
                                                       self.conv5_6_3_1)
        learner57, mask57 = self.cal_learner_and_prior(residual5_3, input5_3, self.conv5_7_1_1, self.conv5_7_2_1,
                                                       self.conv5_7_3_1)
        learner58, mask58 = self.cal_learner_and_prior(residual5_3, input5_3, self.conv5_8_1_1, self.conv5_8_2_1,
                                                       self.conv5_8_3_1)

        l_learner3 = [learner51,learner52,learner53,learner54,learner55,learner56,learner57,learner58]

        out_middle3=0
        for i in range(len(l_learner3)):
            out_middle3+=l_learner3[i]
        #
        l_mask=[mask51,mask52,mask53,mask54,mask55,mask56,mask57,mask58]
        out_middle3=learner51



        return out_middle3,l_learner3,l_mask
    def cal_learner_and_prior(self,residual,input,conv1, conv2,conv3):
        learner = conv1(residual)
        learner = conv3(learner)
        mask = conv2(learner)
        mask=self.sigmoid(mask)
        learner = residual * mask
        learner=learner+input
        return learner, mask


class Gater(nn.Module):
    def __init__(self,num_class):
        super(Gater, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # 卷积名字第一个参数代表第几个模型，第二个参数代表模型中第几个卷积层，第三个是卷积核大小
        self.conv3_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_4 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_5 = BaseConv(512, num_class, 3, 1, activation=None, use_bn=True)

        self.conv9_1= BaseConv(512*9, 4400, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv9_2 = BaseConv(4400, 4200, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv10_1 = BaseConv(4200, 512*8, 1, 1, activation=None, use_bn=True)
    def cal_weight(self,input, conv, prior):
        weight = conv(input)  # 门网络得到的特征图
        # weight=torch.abs(weight)
        weight = (weight + prior) / 2  # 加上先验特征图并求平均
        return weight

    def forward(self,l_prior,input):
        input1=input
        pred_class = self.conv3_5(input1)
        pred_class = torch.nn.functional.adaptive_avg_pool2d(pred_class, (1, 1)).cuda()
        pred_class=torch.nn.Softmax(dim=1)(pred_class)
        pred_class=torch.squeeze(pred_class)

        input = self.conv3_3(input)
        input = self.conv3_4(input)
        #主網絡給門控網絡提供先驗信息
        length1=len(l_prior)
        prior=l_prior[0]
        for i in range(1,length1):
            prior=torch.cat((prior,l_prior[i]),1)

        f_cat=torch.cat((input,prior),1)
        g1=self.conv9_1(f_cat)
        g1=self.conv9_2(g1)
        g1=self.conv10_1(g1)#8192个通道
        g1=torch.nn.functional.adaptive_avg_pool2d(g1,(1,1)).double().cuda()
        mask6 = g1.gt(0)
        g=mask6
        return g,pred_class
class Choice(nn.Module):
    def __init__(self):
        super(Choice,self).__init__()
    def forward(self, l_learner,g):
        g=g.float().cuda()
        model_num = 8
        flag1=g[:,0]#用来判断是训练还是验证过程
        if len(flag1)>1:
            learner_gate=[0]*model_num
            learner_sum=0
            for i in range(model_num):
                learner_gate[i]=self.tensor_multiply(l_learner[i],g,i*512,(i+1)*512)
                learner_sum+=learner_gate[i]
            out=learner_sum
            return out
        else:
            g=torch.squeeze(g)
            learner_gate=[0]*model_num
            learner_sum=0
            for i in range(model_num):
                g_i=g[i*512:(i+1)*512]
                g_i = g_i.unsqueeze(1)
                g_i = g_i.unsqueeze(2)
                learner_gate[i]=l_learner[i]*g_i
                learner_sum+=learner_gate[i]

            out=learner_sum
            return out
    def tensor_multiply(self,learner, g, index1,index2):
        g_0=g[0, index1:index2]
        g_0 = g_0.unsqueeze(1)
        g_0 = g_0.unsqueeze(2)
        learner_0 = learner[0] * g_0
        learner_0 = learner_0.unsqueeze(0)
        learner_gate = learner_0
        for i in range(1, 8):
            g_i = g[i, index1:index2]
            g_i = g_i.unsqueeze(1)
            g_i = g_i.unsqueeze(2)
            learner_temp = learner[i] * g_i
            learner_temp = learner_temp.unsqueeze(0)
            learner_gate = torch.cat((learner_gate, learner_temp), 0)
        return learner_gate

class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        # for p in self.parameters():
        #     p.requires_grad=False

    def forward(self,out,*input):#*,*input  out,
        conv2_2, conv3_3, conv4_3, conv5_3 = input
        input=self.upsample(out)
        # input = self.upsample(conv5_3)
        input = torch.cat([input, conv4_3], 1)#8,3,400,400
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.upsample(input)

        input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        # input = self.upsample(input)
        input = self.conv7(input)

        return input

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        # if self.use_in:
        #     input = self.IN(input)

        if self.activation:
            input = self.activation(input)

        return input
class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

if __name__ == '__main__':
    input = torch.randn(4, 3, 400, 400).cuda()
    model = SFAoatt().cuda()
    output = model(input)
    print(input.size())
    print(output.size())

