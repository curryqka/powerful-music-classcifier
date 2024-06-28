import os
import numpy as np
import torch.nn as nn
import torch
import librosa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from joblib import dump, load

from exp.exp_basic import Exp_Basic
from models.classifier_GTZAN import MelSpecClassifier, GoogleNet, EarlyStopping
from models.classifier_GTZAN import create_mlp_classifier, create_svm_classifier, create_xgb_classifier
from data.data_loader import data_provider, data_provider_ML
from data.data_extractor import get_multi_feat, get_single_feat
from utlis.metric import get_roc_auc_score, show_confusion_matrix, show_results, show_roc, visual_loss
from utlis.check_utlis import check_folder

class EXP_En():
        
    def __init__(self, args, setting):
        
        self.args = args  
        self.device = self._acquire_device()
        self.prepare_val(setting=setting)
        self.prepare_train(setting=setting)
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _get_data(self, flag):


        data_set, data_loader_ML = data_provider_ML(self.args, flag)
        data_set, data_loader_DL = data_provider(self.args, flag)
        return data_loader_ML, data_loader_DL

    def _build_model(self, model_name, setting):
        

        # get ML model
        clf_svm = create_svm_classifier()
        clf_mlp = create_mlp_classifier()
        clf_xgb = create_xgb_classifier()

        model_dict_ML = {
            'svm': clf_svm,
            'mlp': clf_mlp,
            'xgb': clf_xgb
        }
        self.model_ML = model_dict_ML[model_name]
        
        path = './models'
        model_file = path + '/' + 'model.joblib'
        loaded_model = load(model_file)
        self.model_ML = loaded_model

        # get DL model
        model_dict_DL = {
            'resnet' : MelSpecClassifier,
            'googlenet': GoogleNet,
            
        }
        torch.backends.cudnn.enabled = False
        # model = model_dict[self.args.model].Model(self.args).float()
        model = model_dict_DL[self.args.model]().float()
        model = model.to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        # path = os.path.join(self.args.checkpoints, setting)
        # best_model_path = path + '/' + 'checkpoint.pth'
        best_model_path = 'results/checkpoints/resnet_GTZAN_DLexp_0_bs_32/checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        self.model_DL = model
        return loaded_model, model
    

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def prepare_train(self, setting):

        # TODO: add branch --ensemble batch===1
        ML_data, DL_data = self._get_data(flag='ensemble_train')
        model_ML,model_DL = self._build_model('svm',setting)
        ML_outputs, DL_outputs, y_values, y = [], [], [], []
        for i,(id_ML, batch_x, batch_y) in enumerate(ML_data):
            # print(id_ML)
            ML_output = model_ML.predict_proba(batch_x)
            ML_outputs.append(ML_output)
           
        model_DL.eval()
        with torch.no_grad():
            for i,(id_DL, batch_x, batch_y) in enumerate(DL_data):
                # print(id_DL)
                batch_x = batch_x.float().to(self.device)
                DL_output = self.model_DL(batch_x)

                # DL_output = torch.nn.functional.softmax(DL_output, dim=1)
                DL_output = DL_output.cpu().numpy()
                DL_outputs.append(DL_output)
                # prob
                y.append(batch_y.cpu().numpy())

                # class id 
                y_value = [np.where(batch_y[batch_id].cpu().numpy() == 1)[0][0] for batch_id in range(batch_y.shape[0])]
                for value in y_value:
                    y_values.append(value)
        ensemble_set = []
        for ML,DL, y_value, y_ in zip(ML_outputs, DL_outputs, y_values, y):

            # ML = np.argmax(ML, axis=1)
            # DL = torch.argmax(DL, axis=1)
            # y_value = torch.argmax(y_, axis=1)
            
            ensemble_set.append([ML, DL, y_value, y_])
        # self.ensemble_set = ensemble_set
        return ensemble_set
    
    def train(self, setting):
        if self.args.ensemble_mode == 'average':
            weight = [0.5, 0.5]
        elif self.args.ensemble_mode == 'opt':
            weight = [0.4, 0.6]
        elif self.args.ensemble_mode == 'acc':
            weight = [0.05, 0.95]
        ensemble_set = self.prepare_train(setting)
        X = [[x1, x2] for x1,x2, _, _ in ensemble_set]
        y = [[y_value, y_]for _, _, y_value, y_ in ensemble_set]

        # clf = LinearRegression()
        # clf.fit(X,y)
        # acc = clf.score(X,y)

        # metric
        
        preds, pred_probs, labels, label_probs = [], [], [], []
        for x1,x2,y_value, y_ in ensemble_set:
            
            pred_prob = x1 * weight[0] + x2 * weight[1]
            pred = np.argmax(pred_prob)

            preds.append(pred)
            labels.append(y_value)

            pred_probs.append(pred_prob)
            label_probs.append(y_)
            
            
        
        
        # calculate the metric from all of the samples
        accuracy = metrics.accuracy_score(labels, preds)
        precision = metrics.precision_score(labels, preds, average='micro')
        recall = metrics.recall_score(labels, preds, average='weighted')
        F1 = metrics.f1_score(labels, preds, average='weighted')

        pred_probs = np.concatenate(pred_probs)
        label_probs = np.concatenate(label_probs)
        roc_auc = get_roc_auc_score(label_probs, pred_probs)
        print(f"The accuracy is : {accuracy}")
        print(f"The precision is : {precision}")
        print(f"The recall is : {recall}")
        print(f"The F1 score is : {F1}")
        print(f"The ROC_AUC is : {roc_auc}")
    
    def prepare_val(self, setting):
        ML_data, DL_data = self._get_data(flag='ensemble_test')
        model_ML,model_DL = self._build_model('svm',setting)
        ML_outputs, DL_outputs, y_values, y = [], [], [], []
        for i,(id_ML, batch_x, batch_y) in enumerate(ML_data):
            # print(id_ML)
            ML_output = model_ML.predict_proba(batch_x)
            ML_outputs.append(ML_output)
            
        model_DL.eval()
        with torch.no_grad():
            for i,(id_DL, batch_x, batch_y) in enumerate(DL_data):
                # print(id_DL)
                batch_x = batch_x.float().to(self.device)
                DL_output = self.model_DL(batch_x)

                # DL_output = torch.nn.functional.softmax(DL_output, dim=1)
                DL_output = DL_output.cpu().numpy()
                DL_outputs.append(DL_output)
                y.append(batch_y.cpu().numpy())
                y_value = [np.where(batch_y[batch_id].cpu().numpy() == 1)[0][0] for batch_id in range(batch_y.shape[0])]
                for value in y_value:
                    y_values.append(value)
        ensemble_set = []
        for ML,DL, y_value, y_ in zip(ML_outputs, DL_outputs, y_values, y):

            # ML = np.argmax(ML, axis=1)
            # DL = torch.argmax(DL, axis=1)
            # y_value = torch.argmax(y_, axis=1)
            
            ensemble_set.append([ML, DL, y_value, y_])
        self.ensemble_set = ensemble_set
        
        return ensemble_set
    
    def validate(self, setting):
        # example : weight = [0.5, 0.5]
        if self.args.ensemble_mode == 'average':
            weight = [0.5, 0.5]
        elif self.args.ensemble_mode == 'opt':
            # weight = [0.6204, 0.3795]
            weight = [0.2130, 0.7870]
        elif self.args.ensemble_mode == 'acc':
            weight = [0.05, 0.95]
        
        ensemble_set = self.prepare_val(setting)
        X = [[x1, x2] for x1,x2, _, _ in ensemble_set]
        y = [[y_value, y_]for _, _, y_value, y_ in ensemble_set]

        # clf = LinearRegression()
        # clf.fit(X,y)
        # acc = clf.score(X,y)

        # metric
        # log
        check_folder(self.args.log_path)
        saver = open(self.args.log_path + '/' + setting + '.txt', "w")
        saver.write("Exp name: ensemble with weight: {} | {}\n".format(weight[0], weight[1]))
        saver.write("The test result save path is {}\n".format(self.args.log_path))
        saver.flush()
        preds, pred_probs, labels, label_probs = [], [], [], []
        for x1,x2,y_value, y_ in ensemble_set:
            
            pred_prob = x1 * weight[0] + x2 * weight[1]
            pred = np.argmax(pred_prob)

            preds.append(pred)
            labels.append(y_value)

            pred_probs.append(pred_prob)
            label_probs.append(y_)
        
        # save confusion matrix
        save_fig_path = './results/' + setting + '/' + 'figs/'
        save_fig_path = check_folder(save_fig_path)
        
        # calculate the metric from all of the samples
        accuracy = metrics.accuracy_score(labels, preds)
        precision = metrics.precision_score(labels, preds, average='micro')
        recall = metrics.recall_score(labels, preds, average='weighted')
        F1 = metrics.f1_score(labels, preds, average='weighted')

        pred_probs = np.concatenate(pred_probs)
        label_probs = np.concatenate(label_probs)
        roc_auc = get_roc_auc_score(label_probs, pred_probs)

        # get confusion matrix
        show_confusion_matrix(labels, preds, name=os.path.join(save_fig_path, 'confusion_matrix_ensemble.png'))
        print(f"The accuracy is : {accuracy}")
        print(f"The precision is : {precision}")
        print(f"The recall is : {recall}")
        print(f"The F1 score is : {F1}")
        print(f"The ROC_AUC is : {roc_auc}")

        # logs for recording
        saver.write(f"The test sample accuracy is:{accuracy}\n")
        saver.write(f"The test sample precision is:{precision}\n")
        saver.write(f"The test sample recall is:{recall}\n")
        saver.write(f"The test sample f1 score is:{F1}\n")
        saver.write(f"The test sample auc roc is:{roc_auc}\n")
        saver.close()
            
        # 定义适应度函数
    
    def evaluate(self, weights):
    
        w1, w2 = weights
        # 归一化权重
        w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)

        preds, labels, pred_probs, label_probs = [], [], [], []
        for x1,x2,y_value, y_ in self.ensemble_set:
            
            pred_prob = x1 * w1 + x2 * w2
            pred = np.argmax(pred_prob)

            preds.append(pred)
            labels.append(y_value)

            pred_probs.append(pred_prob)
            label_probs.append(y_)
        
        # 计算准确率
        accuracy = metrics.accuracy_score(labels, preds)
        return (accuracy,)

    def test(self, setting, load=False, files=None):

        '''
        This function is used to test the model.
        '''
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()

        # QKA 20240604
        """
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        """
        best_model_path = "./results/checkpoints/resnet_GTZAN_DLexp_0_bs_8/checkpoint.pth"
        self.model_DL.load_state_dict(torch.load(best_model_path))
        # Make predictions using model
        preds = []
        probs = []
        true_values = []
        loss = 0

        # metric
        total_acc, total_pre, total_recall, total_F1 = [], [], [], []
        self.model_DL.eval()  # prep model for evaluation

        # log
        check_folder(self.args.log_path)
        saver = open(self.args.log_path + '/' + setting + '_ensemble.txt', "w")
        saver.write("The test result save path is {}\n".format(self.args.log_path))
        saver.flush()
        with torch.no_grad():
            for i, (_, x_batch, y_batch) in enumerate(test_loader):
                # move to device
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # Make predictions
                pred = self.model_DL(x_batch)
                loss += criterion(pred, y_batch)

                # prob = torch.sigmoid(pred)
                prob = pred
                probs.append(prob.cpu().numpy())
                pred_value = np.argmax(prob.cpu().numpy(), axis=1)
                preds.append(pred_value)
                true_values.append(y_batch.cpu().numpy())

                label = [np.where(y_batch[batch_id].cpu().numpy() == 1)[0][0] for batch_id in range(y_batch.shape[0])]
                
                # cal the metric
                acc = sum(np.array(pred_value) == np.array(label))/len(label)
                pre = metrics.precision_score(label, pred_value, average='micro')
                recall = metrics.recall_score(label, pred_value, average='weighted')
                f1 = metrics.f1_score(label, pred_value, average='weighted')

                total_acc.append(acc)
                total_pre.append(pre)
                total_recall.append(recall)
                total_F1.append(f1)
            #Calculate Accuracy
            # labels = [np.where(true_value==1)[0][0] for true_value in true_values]
            # accuracy = sum(np.array(preds) == np.array(labels))/len(labels)
            probs = np.concatenate(probs)
            preds = np.concatenate(preds)
            true_values = np.concatenate(true_values)
            roc_auc = get_roc_auc_score(true_values, probs)
           
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
    
    def predict(self, setting, files):
        """
        predict the test set
        """
        if not os.path.exists('./models/save_model.joblib'):
            print('No model file exists!')
            exit()
        else:
            
            path = os.path.join(self.args.checkpoints, setting)
            model_file = path + '/' + 'model.joblib'
            loaded_model = load(model_file)

        if loaded_model is None:
            print('No model file exists!')
            exit()

        if files is None:
            print('No test files!')
            exit()

        X_test = get_single_feat(files)

        # distribution of the genre  
        pred = loaded_model.predict([X_test])

        
        return pred
