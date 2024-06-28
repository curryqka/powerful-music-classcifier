from data import data_loader
from exp.exp_basic import Exp_Basic

from models.classifier_GTZAN import MelSpecClassifier, GoogleNet, EarlyStopping
from utlis.metric import visual_loss, visual_lr, show_confusion_matrix, show_results, show_roc, get_roc_auc_score
from sklearn import metrics
from utlis.check_utlis import check_device, check_folder

from utlis.tool import adjust_learning_rate, adjust_teacher_learning_rate

from data.data_extractor import get_multi_feat_CNN,get_single_feat_CNN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        """
        This function is to build the model
        Return:
            The model
        """
        model_dict = {
            'resnet' : MelSpecClassifier,
            'googlenet': GoogleNet,
            
        }
        torch.backends.cudnn.enabled = False
        # model = model_dict[self.args.model].Model(self.args).float()
        model = model_dict[self.args.model]().float()
        model = model.to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, dataloader = data_loader.data_provider(self.args, flag)
        return data_set, dataloader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()
        return criterion

    '''
    This function evaluate the model
    Args: vali_loader
    '''
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (_, batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
                preds.append(pred.cpu().numpy())
                trues.append(true.cpu().numpy())
        total_loss = np.average(total_loss)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        roc_auc = get_roc_auc_score(trues, preds)
        self.model.train()
        return total_loss, roc_auc 


    '''
    This function is used to train the model.

    Args: setting parameter
    
    '''
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='test')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # save train_loss,vail_loss and test_loss for every epoch
        save_fig_path = './results/' + setting + '/' + 'figs/'
        save_log_path = './results/' + setting + '/' + 'logs/'
        save_fig_path = check_folder(save_fig_path)
        save_log_path = check_folder(save_log_path)

        save_train_loss = []
        save_vali_loss = []
        save_test_loss = []
        saver = open(os.path.join(save_log_path, 'logs.txt'), "w")
        saver.write("The log file save path is {}\n".format(save_log_path))
        saver.flush()

        # train loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (_, batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
        
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # 能用的数据送到模型
                        outputs = self.model(batch_x)
                        # 计算损失函数
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    # outputs = torch.sigmoid(outputs)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    with torch.autograd.detect_anomaly():
                        # 梯度反传
                        loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, _ = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {} train loss:{:.4f} | val loss:{:.4f}".format(epoch + 1, train_loss, vali_loss))

            save_train_loss.append(train_loss.item())
            save_vali_loss.append(vali_loss.item())
            
          

            saver.write("Epoch: {} cost time: {}\n".format(epoch + 1, time.time() - epoch_time))
            saver.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} \n".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            saver.flush()


            # training strategic
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                saver.write("Early stopping")
                saver.flush()
                
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        saver.write("The best model is saved at {}\n".format(best_model_path))
        saver.flush()
        saver.close()

        visual_loss(train_loss=save_train_loss, vali_loss=save_vali_loss, name=os.path.join(save_fig_path, 'loss.png'))
        return self.model

    def test(self, setting, load=False, files=None):

        '''
        This function is used to test the model.
        '''
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()

        # QKA 20240604
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # Make predictions using model
        preds = []
        probs = []
        true_values = []
        labels = []
        loss = 0

        # metric
        total_acc, total_pre, total_recall, total_F1 = [], [], [], []
        self.model.eval()  # prep model for evaluation

        # 

        # save train_loss,vail_loss and test_loss for every epoch
        save_fig_path = './results/' + setting + '/' + 'figs'
        save_fig_path = check_folder(save_fig_path)

        # log
        check_folder(self.args.log_path)
        saver = open(self.args.log_path + '/' + setting + '.txt', "w")
        saver.write("The test result save path is {}\n".format(self.args.log_path))
        saver.flush()
        with torch.no_grad():
            for i, (_, x_batch, y_batch) in enumerate(test_loader):
                # move to device
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # Make predictions
                pred = self.model(x_batch)
                loss += criterion(pred, y_batch)

                # prob = torch.sigmoid(pred)
                prob = pred
                probs.append(prob.cpu().numpy())
                # pred_value = np.argmax(F.softmax(prob, dim=1).cpu().numpy(), axis=1)
                pred_value = np.argmax(prob.cpu().numpy(), axis=1)
                preds.append(pred_value)
                true_values.append(y_batch.cpu().numpy())

                label = [np.where(y_batch[batch_id].cpu().numpy() == 1)[0][0] for batch_id in range(y_batch.shape[0])]
                labels.append(label)
                # cal the metric
                acc = sum(np.array(pred_value) == np.array(label))/len(label)
                pre = metrics.precision_score(label, pred_value, average='macro')
                recall = metrics.recall_score(label, pred_value, average='macro')
                f1 = metrics.f1_score(label, pred_value, average='weighted')

                total_acc.append(acc)
                total_pre.append(pre)
                total_recall.append(recall)
                total_F1.append(f1)
            #Calculate Accuracy
            # labels = [np.where(true_value==1)[0][0] for true_value in true_values]
            # accuracy = sum(np.array(preds) == np.array(labels))/len(labels)
            probs = np.concatenate(probs)
            
            true_values = np.concatenate(true_values)
            roc_auc = get_roc_auc_score(true_values, probs)

            # get confusion matrix
            show_confusion_matrix(labels, preds, name=os.path.join(save_fig_path, 'confusion_matrix.png'))
            # calculate the metric from all of the samples
            accuracy = np.mean(total_acc)
            precision = np.mean(total_pre)
            recall = np.mean(total_recall)
            F1 = np.mean(total_F1)
            
            # logs for recording
            saver.write(f"The test sample accuracy is:{accuracy}\n")
            saver.write(f"The test sample precision is:{precision}\n")
            saver.write(f"The test sample recall is:{recall}\n")
            saver.write(f"The test sample f1 score is:{F1}\n")
            saver.write(f"The test sample auc roc is:{roc_auc}\n")
            saver.close()
            print(f"The test sample accuracy is:{accuracy}\n")
            print(f"The test sample auc roc is:{roc_auc}\n")
        return preds, true_values, accuracy, precision, recall, F1, roc_auc


    def _get_untrained_well_sample(self, setting):
        train_data, train_loader = self._get_data(flag='eval_train')
        criterion = self._select_criterion()
        total_loss = []
        all_sample = []
        
        with torch.no_grad():
            for i, (_, x_batch, y_batch) in enumerate(train_loader):
                # move to device
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # Make predictions
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)

                # data_dict
                data_dict = dict(
                    id=i,
                    loss=loss
                )
                total_loss.append(loss.item())
                all_sample.append(data_dict)
        avg_loss = np.average(total_loss)
        num_untrained_well_sample = 0
        with open(os.path.join(self.args.save_path, 'VideoSets', 'TrainSet_ML.txt'), 'w') as f:
            for dd in all_sample:
                sample_loss = dd['loss']

                # save those are not trained well
                if sample_loss >= avg_loss:
                    num_untrained_well_sample += 1
                    sample_id = dd['id']
                    f.write(f"{sample_id} {sample_loss}\n")
            
        print(f"The number of selected sample: {num_untrained_well_sample}")
    def predict(self, setting, load=True, files=None):
        '''
        This function is used to predict the genre.

        '''
        pred_data, pred_loader = self._get_data(flag='test')
        criterion = self._select_criterion()
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        mel_, feature, length_ = get_single_feat_CNN(files)
        # print(mel_)
        mel_ = mel_[np.newaxis,...]
        pred_prob = self.model(torch.Tensor(mel_).float().to(self.device))
        pred_prob = pred_prob.detach().cpu().numpy()
        # pred = []
        # for i in range(length_):
        #     pred.append(self.model(torch.Tensor([feature[i]]).float().to(self.device)))
        
        # print(pred)
        # ans = pred[0]
        # for i in range(length_ -1):
        #     ans = ans + pred[i+1]
        
        #print(ans)
        # np.argmax(ans.tolist())

        def normalization(data):
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

        # data = normalization(ans.tolist())
        # print(data[0])

        pred_prob = normalization(pred_prob)
        print(pred_prob.shape)

        pred_prob_list = [pred_prob[0,i] for i in range(pred_prob.shape[1])]
        x_data =['blues',
            'classical',
            'country',
            'disco',
            'hiphop',
            'jazz',
            'metal',
            'pop',
            'reggae',
            'rock']
        # y_data = data[0]
        y_data = pred_prob_list

        plt.figure(figsize=(10, 6))
        for i in range(len(x_data)):
            plt.bar(x_data[i], y_data[i])

        plt.title("show results")
        # x
        plt.xlabel("genres")
        # y
        plt.ylabel("scores")
        # 防止x轴标签重叠，可以旋转标签
        plt.xticks(rotation=45)
        plt.savefig('results/predict.png')
        # plt.show()
        

        return None
