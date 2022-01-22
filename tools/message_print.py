def print_summary(exp_name, scores, train_record):
    mae, mse, loss = scores
    # print('=' * 50)
    print(exp_name)
    # print('    ' + '-' * 20)
    print('[mae %.2f mse %.2f], [val loss %.4f]' % (mae, mse, loss))
    # print('    ' + '-' * 20)
    print('[best] [model: %s] , [mae %.2f], [mse %.2f]' % (
        train_record['best_model_name'], train_record['best_mae'], train_record['best_mse']))
    # print('=' * 50)
