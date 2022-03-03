from easydict import EasyDict as edict

# init
__C_QNRF = edict()
cfg_data = __C_QNRF
__C_QNRF.TRAIN_SIZE = (400, 400)
__C_QNRF.DATA_PATH = 'datasets/QNRF/processed_data_size_15_4'
# __C_QNRF.DATA_PATH = '/media/wmy/ee25778f-7952-4c55-9d90-e4337ff23c7d/wmy/crowd_counting/QNRF/processed_data_size_15_4'
__C_QNRF.MEAN_STD = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])
__C_QNRF.log_para = 100
__C_QNRF.label_factor = 2
