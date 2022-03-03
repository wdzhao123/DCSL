from easydict import EasyDict as edict

# init
__C_SHB = edict()
cfg_data = __C_SHB
__C_SHB.TRAIN_SIZE = (400, 400)
__C_SHB.DATA_PATH = '/media/wmy/c9792eb3-1646-c646-81c7-34924646579b/wmy/CNNProjects/crowd-code/datasets/SHB/processed_data_size_15_4'
__C_SHB.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],
                    [0.23242045939, 0.224925786257, 0.221840232611])
__C_SHB.log_para = 100
__C_SHB.label_factor = 2
