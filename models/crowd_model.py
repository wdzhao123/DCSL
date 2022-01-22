import time
import torch.nn as nn
from network.loss import *
from tensorboardX import SummaryWriter

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name, if_vector_loss=False, use_my_loss=False, gt_log_para=100):
        super(CrowdCounter, self).__init__()
        self.net_name = model_name
        self.gt_log_para = gt_log_para
        from models.backbones.model import SFAoatt as net
        self.CCN = net()

        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()

        self.loss_mse_fn = nn.MSELoss().cuda()

        self.use_my_loss = use_my_loss

        self.my_loss_mse_fn = wmyloss_cos().cuda()

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_map, epoch, gt_adaptive=None):
        density_map, list_learner,pred_class= self.CCN(img)#
        self.loss_mse = self.build_loss(density_map=density_map.squeeze(), gt_data=gt_map.squeeze(),list_learner=list_learner,pred_class=pred_class
                                        )
        density_map = torch.mean(density_map,dim=-3)
        return density_map

    def build_loss(self, density_map=None,
                   gt_data=None,
                   list_learner=None,pred_class=None):

        loss_mse = self.my_loss_mse_fn(density_map, gt_data, list_learner,pred_class)
        return loss_mse

    def test_forward(self, img):
        density_map, list_learner,pred_class = self.CCN(img)
        return density_map, list_learner ,pred_class




