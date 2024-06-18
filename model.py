# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:29:56 2020
@author: Youngdo Ahn
"""
import torch.nn as nn  
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class BaseModel(nn.Module):
    def __init__(self, input_dim=1582, last_dim=512, AE_task=False, ADV_task=False, CLT_task=False):
        super(BaseModel, self).__init__()
        self.CLT_task = CLT_task
        self.ADV_task = ADV_task
        self.AE_task  = AE_task

        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(input_dim, 1024)) #  1582
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_drop1', nn.Dropout())
        
        self.feature.add_module('f_fc2', nn.Linear(1024, 1024))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_drop2', nn.Dropout())
        
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(1024, last_dim)) # 1024, 512
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(last_dim, last_dim))
        self.class_classifier.add_module('c_drop2', nn.Dropout())
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        
        self.class_classifier_last = nn.Sequential()
        self.class_classifier_last.add_module('c_fc3', nn.Linear(last_dim, 4))
        #self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        self.class_classifier_last = nn.Sequential()
        self.class_classifier_last.add_module('c_fc3', nn.Linear(last_dim, 4))
        
        if self.CLT_task:
            self.clt_classifier = nn.Sequential()
            self.clt_classifier.add_module('t_fc1', nn.Linear(1024, last_dim)) # 1024, 512
            self.clt_classifier.add_module('t_drop1', nn.Dropout())
            self.clt_classifier.add_module('t_relu1', nn.ReLU(True))
            self.clt_classifier.add_module('t_fc2', nn.Linear(last_dim, last_dim))
            self.clt_classifier.add_module('t_drop2', nn.Dropout())
            self.clt_classifier.add_module('t_relu2', nn.ReLU(True))
            self.clt_classifier.add_module('t_fc3', nn.Linear(last_dim, self.CLT_task))
        if self.AE_task:
            self.feature_g = nn.Sequential()
            self.feature_g.add_module('fg_fc1', nn.Linear(1024, 1024))
            self.feature_g.add_module('fg_dr1', nn.Dropout())
            self.feature_g.add_module('fg_fc2', nn.Linear(1024, input_dim))

        
    def forward(self, input_data):
        input_data = input_data.view(input_data.data.shape[0], input_data.data.shape[1]) # 1582 512
        feature = self.feature(input_data)
        feature = feature.view(input_data.data.shape[0],1024) #feature.view(-1, 50 * 4 * 4)
        #class_output = self.class_classifier(feature)
        e_pr_output = self.class_classifier(feature)
        #e_pr_output = nn.functional.normalize(e_pr_output)
        class_output = self.class_classifier_last(e_pr_output)
        if self.AE_task:
            g_feature = self.feature_g(feature).view(input_data.data.shape[0], input_data.data.shape[1]) 
        if self.CLT_task:
            clt_output = self.clt_classifier(feature)
        if self.AE_task and self.CLT_task:
            return class_output, e_pr_output, clt_output, g_feature
        elif self.CLT_task:
            return class_output, e_pr_output, clt_output
        elif self.AE_task:
            return class_output, e_pr_output, g_feature
        else:
            return class_output, e_pr_output # feature#
        

#if self.ADV_task:
#    feature = ReverseLayerF.apply(feature, 0.5)

