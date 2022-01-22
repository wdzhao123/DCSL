# -*-coding:utf-8-*-
import os


def logger(exp_path, exp_name, data_mode):
    from tensorboardX import SummaryWriter
    if not os.path.exists('./exp_outputs'):
        os.mkdir('./exp_outputs')
    # 写配置文件到日志中
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path + '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    # 读不了yaml则删
    cfg_file_dir = './exp_configs' + '/' + exp_name.split('_')[-2]\
                   + '_' + exp_name.split('_')[-1] + '/' + 'config.yaml'

    cfg_file = open(cfg_file_dir, "r")
    cfg_lines = cfg_file.readlines()
    with open(log_file, 'a') as f:
        f.write('#This is config file' + '\n\n')
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    # if data_mode == 'SHB':
    #     date_setting_path = 'datasets/SHB/base_cfg.py'
    # elif data_mode == 'SHA':
    #     date_setting_path = 'datasets/SHA/base_cfg.py'
    # elif data_mode == 'BGI':
    #     date_setting_path = 'datasets/BGI/base_cfg.py'
    # elif data_mode == 'QNRF':
    #     date_setting_path = 'datasets/QNRF/base_cfg.py'
    date_setting_path = 'datasets/SHB/base_cfg.py'


    data_setting_file = open(date_setting_path, "r")
    data_setting_lines = data_setting_file.readlines()
    with open(log_file, 'a') as f:
        f.write('#This is data setting file' + '\n\n')
    with open(log_file, 'a') as f:
        f.write(''.join(data_setting_lines) + '\n\n\n\n')

    return writer, log_file


def logger_txt(log_file, epoch, scores):
    mae, mse, loss = scores

    snapshot_name = 'all_ep_%d_mae_%.1f_mse_%.1f' % (epoch + 1, mae, mse)

    with open(log_file, 'a') as f:
        f.write('=' * 15 + '+' * 15 + '=' * 15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('    [mae %.2f mse %.2f], [val loss %.4f]\n' % (mae, mse, loss))
        f.write('=' * 15 + '+' * 15 + '=' * 15 + '\n\n')
