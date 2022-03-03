from easydict import EasyDict as edict

# init
__C_UCF50 = edict()
cfg_data = __C_UCF50
__C_UCF50.TRAIN_SIZE = (400, 400)
__C_UCF50.DATA_PATH = 'datasets/UCF50/processed_data_size_15_4'
__C_UCF50.MEAN_STD = (
[0.403584420681, 0.403584420681, 0.403584420681], [0.268462955952, 0.268462955952, 0.268462955952])
__C_UCF50.log_para = 100
__C_UCF50.label_factor = 2

__C_UCF50.VAL_INDEX = 3
