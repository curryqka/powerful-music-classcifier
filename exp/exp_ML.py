import os
import numpy as np
import torch.nn as nn
import torch
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump, load


from models.classifier_GTZAN import create_mlp_classifier, create_svm_classifier, create_xgb_classifier
from data.data_loader import data_provider_ML
from data.data_extractor import get_multi_feat, get_single_feat
from utlis.metric import get_roc_auc_score, show_confusion_matrix, show_results, show_roc, visual_loss
from utlis.check_utlis import check_folder

class EXP_ML:
    def __init__(self, args):
        self.args = args

    def _get_data(self, flag):

        data_set, data_loader = data_provider_ML(self.args, flag)
        return data_set, data_loader

    def _get_model(self, model_name):
        
        clf_svm = create_svm_classifier()
        clf_mlp = create_mlp_classifier()
        clf_xgb = create_xgb_classifier()

        model_dict = {
            'svm': clf_svm,
            'mlp': clf_mlp,
            'xgb': clf_xgb
        }
        self.model = model_dict[model_name]

        return model_dict[model_name]
    
    def _select_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        return criterion
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='test')
        
        # select the model to train
        # model = self._get_model('svm')
        model = self._get_model(self.args.model)
        # select the criterion
        criterion = self._select_criterion()

        # save train_loss,vail_loss and test_loss for every epoch
        save_fig_path = './results/' + setting + '/' + 'figs/'
       
        save_fig_path = check_folder(save_fig_path)

        acc = -1e4

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            for i,(_, batch_x, batch_y) in enumerate(train_loader):
                
                
                model.fit(batch_x, batch_y)
                outputs = model.predict(batch_x)
                # outputs_tensor = torch.from_numpy(outputs).long()
                # loss = criterion(batch_y, outputs_tensor)
                # train_loss.append(loss)
                
                iter_count += 1
                
                
                # accuracy = model.score(X_test, y_test)
                if (i + 1) % 10 == 0:
                    print("Epoch: %d, Iteration: %d, Loss: %.5f" % (epoch, iter_count, 0))
                    iter_count = 0

            train_loss = np.average(train_loss)
            vali_loss, accuracy, roc_auc, total_pre, total_recall, total_F1 = self.validate(vali_data, vali_loader, model, setting)
            
            print("Epoch: %d, Train Loss: %.5f, Validation Loss: %.5f, Validation Accuracy: %.5f, ROC_AUC: %.5f, \
                  Precision: %.5f, Recall: %.5f, F1: %.5f \n" % \
                  (epoch, train_loss, vali_loss, accuracy, roc_auc, total_pre, total_recall, total_F1))

            if acc < accuracy:
                dump(model, './models/model.joblib')
                acc = accuracy
                print("The bestModel saved!")

        # visual_loss(train_loss, vali_loss, name=os.path.join(save_fig_path, 'loss.png'))     
    
    def validate(self, vai_data, vali_loader, model, setting):

        total_loss = []
        probs = []
        trues = []
        preds = []
        labels = []
        accuracy = []
        total_pre, total_recall, total_F1 = [], [], []

        save_fig_path = './results/' + setting + '/' + 'figs/'
        save_fig_path = check_folder(save_fig_path)

        check_folder(self.args.log_path)

        saver = open(os.path.join(self.args.log_path, 'result.txt'), 'w')
        saver.write(f"The model name is {self.args.model}\n")
        for i, (_, batch_x, batch_y) in enumerate(vali_loader):
           
            pred_value = model.predict(batch_x)
            for item in pred_value.tolist():
                preds.append(item)

            label = batch_y.detach().cpu().numpy()
            # class num = 10
            class_num = len(set(label))
            true_value = np.ones([batch_y.shape[0], class_num])
            onehot_cls = np.zeros([class_num])
            for j in range(batch_y.shape[0]):
                onehot_cls = np.zeros([class_num])
                onehot_cls[batch_y[j]]=1
                true_value[j, :] = onehot_cls


            prob = model.predict_proba(batch_x)
            # loss = criterion(pred, true)

            acc = model.score(batch_x, batch_y)
            # cal the metric
            for item in label.tolist():

                labels.append(item)

            # print(label, pred)
            pre = metrics.precision_score(label,  pred_value, average='macro')
            
            recall = metrics.recall_score(label,  pred_value, average='weighted')
            f1 = metrics.f1_score(label,  pred_value, average='weighted')
            # total_loss.append(loss)
            probs.append(prob)
            trues.append(true_value)
            accuracy.append(acc)
            total_pre.append(pre)
            total_recall.append(recall)
            total_F1.append(f1)

        total_loss = np.average(total_loss)
        probs = np.concatenate(probs)
        trues = np.concatenate(trues)
        roc_auc = get_roc_auc_score(trues, probs)

        # get confusion matrix
        show_confusion_matrix(labels, preds, name=os.path.join(save_fig_path, 'confusion_matrix.png'))
        accuracy = np.average(accuracy)
        total_pre = np.average(total_pre)
        total_recall = np.average(total_recall)
        total_F1 = np.average(total_F1)
        # add roc_auc_score
        saver.write(f"The accuracy is : {accuracy}\n")
        saver.write(f"The precision is : {total_pre}\n")
        saver.write(f"The recall is : {total_recall}\n")
        saver.write(f"The F1 score is : {total_F1}\n")
        saver.write(f"The ROC_AUC is : {roc_auc}\n")
        saver.close()
        return total_loss, accuracy, roc_auc, total_pre, total_recall, total_F1
    

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
