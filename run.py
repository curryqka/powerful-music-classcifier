import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from torch.nn.functional import softmax
from deap import base, creator, tools, algorithms

import argparse
import os
import torch
from exp.exp_main import Exp_Main
from exp.exp_ensemble import EXP_En
from utlis.metric import visual_GA
import random
import numpy as np


def main():
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    parser = argparse.ArgumentParser(description="GTZAN-DL-EXP")
    # basic config
    # model
    parser.add_argument('--model', type=str, default='resnet', help='the model type [googlenet, resnet], etc.')
    parser.add_argument('--ML_model', type=str, default='xgb', help='the model type [xgb, svm, mlp], etc.')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--log_path', type=str, default='./logs', help='the path of log')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ensemble_mode', type=str, default='average', help='the ensemble method [average/opt/train roc auc]')
    parser.add_argument('--only_args', default=False, type=str, help='whether to choose only args')
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
    parser.add_argument('--pred_path', type=str, default=r'D:\学生工作\AI项目\music-genre-GTZAN\music-main\data\archive\Data\genres_original\blues\blues.00097.wav', help='location of data tp make predictions')

    # 数据文件名
    parser.add_argument('--data_filename', type=str, default='data.txt', help='data filename')

    # 预训练模型的文件路径
    parser.add_argument('--pretrained_model_path', type=str, default='/path/to/pretrained/model', help='预训练模型的文件路径')
    parser.add_argument('--ensemble_models', type=str, default='', help='pretrained models')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='CE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU  # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # args.use_gpu = False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.anylearn == 1:
        args.root_path = os.path.join(args.dataset_path, 'ETT-small')

    ensemble = args.ensemble
    only_args = args.only_args
    print('Args in experiment:')
    print(args)

    Exp = Exp_Main
    if not ensemble:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_DLexp_{}_bs_{}'.format(
                args.model,
                args.dataset, ii,
                args.batch_size)
            
            # not train test pred
            exp = Exp(args)  # set experiments
            # '''
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            print('>>>>>>>eval which sample are not trained well : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp._get_untrained_well_sample(setting)
            # '''
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True, args.pred_path)
        torch.cuda.empty_cache()
    elif ensemble:
        if only_args:
            return args
        else:
            Exp = EXP_En
            for ii in range(args.itr):
                # setting record of experiments
                setting = 'ensemble_DL_{}_ML_{}_{}_exp{}'.format(
                    args.model, args.ML_model,
                    args.dataset, ii)
                
                # not train test pred
                exp = Exp(args, setting=setting)  # set experiments
                # '''
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing origin: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.test(setting)


                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.validate(setting)
            torch.cuda.empty_cache()

    return args
def run_GA(args, Exp:EXP_En=EXP_En):
    setting = 'ensemble_DL_{}_ML_{}_{}_exp{}'.format(
                args.model, args.ML_model,
                args.dataset, 0)
    exp = Exp(args, setting)
    # 创建遗传算法工具箱
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.rand)  # 初始化权重为随机浮点数
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: [np.random.rand(), 1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def normalize_weights(individual):
        total = sum(individual)
        return [w / total for w in individual]

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", exp.evaluate)

    # 初始化种群
    population = toolbox.population(n=20)

    # 遗传算法参数
    NGEN = 50
    CXPB = 0.5
    MUTPB = 0.2

    # 记录适应度和权重的变化
    fitness_history = []
    weights_history = []
    population_history = []
    # 运行遗传算法
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        
        # 标准化权重
        for ind in offspring:
            ind[:] = normalize_weights(ind)
        
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        
        # 记录当前代的最优个体
        best_ind = tools.selBest(population, 1)[0]
        fitness_history.append(best_ind.fitness.values[0])
        weights_history.append(best_ind[:])
        population_history.append([ind[:] for ind in population])  # 记录种群的位置
        print(f"Generation {gen}: Best accuracy = {best_ind.fitness.values[0]}, Weights = {best_ind}")

    # 输出最终结果
    best_ind = tools.selBest(population, 1)[0]
    print(f"Final Best accuracy = {best_ind.fitness.values[0]}, Weights = {best_ind}")

    visual_GA(weights_history=weights_history, fitness_history=fitness_history, population_history=population_history, \
              NGEN=NGEN, evaluate=exp.evaluate)
if __name__ == "__main__":
    args = main()
    run_GA(args=args)

