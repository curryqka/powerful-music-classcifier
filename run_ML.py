import argparse
import os
import torch
from exp.exp_ML import EXP_ML
import random
import numpy as np

def main():
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    parser = argparse.ArgumentParser(description="GTZAN-ML-EXP")
    # basic config
    # 模型类别
    parser.add_argument('--model', type=str, default='svm', help='the model type [svm, mlp, xgb], etc.')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--log_path', type=str, default='./logs_ML', help='the path for logs')
    # 数据集
    parser.add_argument('--dataset', type=str, default='GTZAN', help='数据集名称, 例如 imagenet, cifar10, etc.')
    parser.add_argument('--save_path', type=str, default='./data', help='data save path')
    
    # data loader
    parser.add_argument('--dataset_path', type=str, default='.', help='root path of the dataset on anylearn')
    parser.add_argument('--anylearn', type=int, default=0, help='whether the task is on anylearn')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')

    # 数据集路径
    parser.add_argument('--data_path', type=str, default='data/archive/Data/genres_original', help='/path/to/dataset')
    parser.add_argument('--checkpoints', type=str, default='./results/checkpoints', help='location of model checkpoints')
    parser.add_argument('--pred_path', type=str, default='./results/predictions', help='location of data tp make predictions')
    
    # 数据文件名
    parser.add_argument('--data_filename', type=str, default='data.txt', help='数据文件的名称')

    # 预训练模型的文件路径
    parser.add_argument('--pretrained_model_path', type=str, default='/path/to/pretrained/model', help='预训练模型的文件路径')
    parser.add_argument('--ensemble_models', type=str, default='', help='pretrained models')
    
    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    args = parser.parse_args()

    if args.anylearn == 1:
        args.root_path = os.path.join(args.dataset_path, 'ETT-small')

    print('Args in experiment:')
    print(args)

    Exp = EXP_ML

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_MLexp_{}'.format(
            args.model,
            args.dataset, ii)
        # not train test pred
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, args.pred_path)
        
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()