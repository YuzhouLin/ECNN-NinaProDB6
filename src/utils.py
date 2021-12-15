import numpy as np
import torch
import torch.nn as nn
import helps_pro as pro
import pandas as pd
import os


class NinaProDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, item):
        return {
            "x": torch.tensor(self.features[item, :], dtype=torch.float),
            "y": torch.tensor(self.targets[item, :], dtype=torch.float)
        }


class EngineTrain:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def criterion(outputs, targets, loss_params):  # loss function
        if loss_params['edl_used'] == 0:
            loss_fun = nn.CrossEntropyLoss()
            loss = loss_fun(outputs, targets)
        else:
            loss = pro.edl_mse_loss(outputs, targets, loss_params)
        return loss

    def train(self, data_loaders, loss_params):
        final_loss = {}
        for phase in ['train', 'val']:
            train_flag = phase == 'train'
            self.model.train() if train_flag else self.model.eval()
            final_loss[phase] = 0.0
            data_n = 0.0
            for _, (inputs, targets) in enumerate(data_loaders[phase]):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)  # (batch_size,)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(train_flag):
                    outputs = self.model(inputs)  # (batch_size,class_n)
                    loss = self.criterion(outputs, targets, loss_params)
                    if train_flag:
                        loss.backward()
                        self.optimizer.step()
                final_loss[phase] += loss.item() * inputs.size(0)
                data_n += inputs.size(0)
            final_loss[phase] = final_loss[phase] / data_n
        return final_loss

    def re_train(self, data_loader, loss_params):
        final_loss = 0.0
        self.model.train()
        data_n = 0.0
        for _, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, loss_params)
                loss.backward()
                self.optimizer.step()
            final_loss += loss.item() * inputs.size(0)
            data_n += inputs.size(0)
        final_loss = final_loss / data_n
        return final_loss


class EngineTest:
    def __init__(self, outputs, targets):
        # outputs: tensor; targets: numpy array
        self.outputs = outputs  # torch size: [1547, 12]
        self.targets = targets[:, np.newaxis]  # [1547, 1]

    def get_output_results(self, acti_fun):  # outputs after activation func
        output_results = \
                eval('pro.' + acti_fun + '_evidence(self.outputs).numpy()')
        return output_results  # numpy array

    def get_pred_results(self):  # prediction results; right or wrong
        preds = self.outputs.argmax(dim=1, keepdim=True).numpy()
        pred_results = preds == self.targets
        return pred_results

    def update_result_acc(self, params):
        # pred: prediction Results (not labels)
        # true: Ground truth labels
        # params: dict -->
        # {'sb_n': , edl', 'outer_f'}

        # load current result file
        filename = 'results/cv/accuracy_temp.csv'
        column_names = [*params, 'gesture', 'recall']
        # 'sb_n','edl_used','outer_f'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=column_names)
        # Update it
        temp_dict = params
        recall = pro.cal_recall(self.outputs, self.targets)
        for class_index, each_recall in enumerate(recall):
            temp_dict['gesture'] = class_index + 1
            temp_dict['recall'] = each_recall
            df = df.append([temp_dict])
        # Save it
        df.to_csv(filename, index=False)
        return

    def update_result_R(self, params):
        # pred: prediction Results (not labels)
        # true: Ground truth labels
        
        # load current result file
        filename = 'results/cv/reliability_temp.csv'
        column_names = [*params, 'skew', 'uncertainty', 'AUROC', 'AP', 'nAP']
        # *params: dict --> 'sb_n','edl_used','outer_f'
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=column_names)

        temp_dict = params
        #  Calculate the scores
        output_results = self.get_output_results(params['acti_fun'])
        scores = pro.cal_scores(output_results, params['edl_used'])

        #  Calculate the labels for misclassification detection
        pred_results = self.get_pred_results()
        labels = np.logical_not(pred_results)

        #  Calculate the AUROC, AP and normalised AP
        #  pm: performance measures
        AUROCs, APs, nAPs, skew = pro.cal_mis_pm(labels, scores)
        #  Update it
        temp_dict['skew'] = skew
        #  Update the AUROC, AP, nAP for each un
        for un_type, AUROC in AUROCs.items():
            temp_dict['uncertainty'] = un_type
            temp_dict['AUROC'] = AUROC
            temp_dict['AP'] = APs[un_type]
            temp_dict['nAP'] = nAPs[un_type]
            df = df.append([temp_dict])
        
        '''
        #  Update the AP for each un
        for un_type, AP in APs.items():
            temp_dict['uncertainty'] = un_type
            temp_dict['AP'] = AP
            df = df.append([temp_dict])

        #  Update the AUROC for each un
        for un_type, AUROC in AUROCs.items():
            temp_dict['uncertainty'] = un_type
            temp_dict['AUROC'] = AUROC
            df = df.append([temp_dict])
        '''
        # Save it
        df.to_csv(filename, index=False)
        return


class Model(nn.Module):
    def __init__(self, number_of_class=12, dropout=0.5, k_c=3):
        # k_c: kernel size of channel
        super(Model, self).__init__()
        # self._batch_norm0 = nn.BatchNorm2d(1)
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(k_c, 5))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(dropout)
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 3))

        self._conv2 = nn.Conv2d(32, 64, kernel_size=(k_c, 5))
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(dropout)
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 3))

        self._fc1 = nn.Linear(26880, 500)
        # 8 = 12 channels - 2 -2 ;  53 = ((500-4)/3-4)/3
        self._batch_norm3 = nn.BatchNorm1d(500)
        self._prelu3 = nn.PReLU(500)
        self._dropout3 = nn.Dropout(dropout)

        self._output = nn.Linear(500, number_of_class)
        self.initialize_weights()

        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        # x = x.permute(0,1,3,2)  --> batch * 1 * 16 * 50
        # print(x.size())
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        # conv1 = self._dropout1(
        # self._prelu1(self._batch_norm1(self._conv1(self._batch_norm0(x)))))
        pool1 = self._pool1(conv1)
        conv2 = self._dropout2(
            self._prelu2(self._batch_norm2(self._conv2(pool1))))
        pool2 = self._pool2(conv2)
        flatten_tensor = pool2.view(pool2.size(0), -1)
        fc1 = self._dropout3(
            self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        output = self._output(fc1)
        return output
