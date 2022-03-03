from easydict import EasyDict as edict

# init
__C_SHA = edict()
cfg_data = __C_SHA
__C_SHA.TRAIN_SIZE = (400,400)
# __C_SHA.TRAIN_SIZE = (2000,2000)
# __C_SHA.DATA_PATH = 'datasets/SHA/processed_data_size_15_4'
__C_SHA.DATA_PATH = '/media/wmy/ee25778f-7952-4c55-9d90-e4337ff23c7d/wmy/crowd_counting/SHA/processed_data_size_15_4'
# __C_SHA.DATA_PATH = '/media/wmy/ee25778f-7952-4c55-9d90-e4337ff23c7d/wmy/project/crowd-code/datasets/SHA_SHB'
__C_SHA.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
# __C_SHA.MEAN_STD = ([0.485, 0.456, 0.406],
#                     [0.229, 0.224, 0.225])
__C_SHA.log_para = 100
__C_SHA.label_factor = 2
