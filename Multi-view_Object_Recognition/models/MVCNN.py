import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
import copy

#mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
#std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

# def flip(x, dim):
#     xsize = x.size()
#     dim = x.dim() + dim if dim < 0 else dim
#     x = x.view(-1, *xsize[dim:])
#     x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
#                       -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
#     return x.view(xsize)


def KL_loss_function(mu1,sigma1,sigma2 = 1):


    batch_size = mu1.size()[0]
    J = mu1.size()[1]

    mu_diff = (mu1) ** 2
    var1 = sigma1 ** 2
    var2 = sigma2 ** 2

    var_frac = var1 / var2
    diff_var_frac = mu_diff / var2

    term1 = torch.sum(torch.log(var_frac)) / batch_size
    term2 = torch.sum(var_frac) / batch_size
    term3 = torch.sum(diff_var_frac) / batch_size

    return - 0.5 * (term1 - term2 -term3 + J)

class SVCNN_IB(Model):

    def __init__(self, name, nclasses=40, pretraining=True):
        super(SVCNN_IB, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        #self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        #self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
            
        self.net_1 = models.vgg11(pretrained=self.pretraining).features
        self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
        self.net_2._modules['6'] = nn.Linear(4096,40)

        self.IB_mu = nn.Sequential(
                                nn.Linear(40, 40),
                                nn.Tanh()
                                )
        self.IB_sigma = nn.Sequential(
                                nn.Linear(40,40),
                                nn.Sigmoid()
                                )

        self.classifier = nn.Sequential(
                                nn.Linear(40,40),
                                nn.ReLU(),
                                nn.Linear(40,40)
                                )


    def IB_feature_extraction(self, x):

        extracted_feature = self.net_1(x)
        extracted_feature = self.net_2(extracted_feature.view(extracted_feature.shape[0],-1))

        mu = 10 * self.IB_mu(extracted_feature) # factor 10 rescales the output range
        sigma = self.IB_sigma(extracted_feature) # represents std

        KL_loss = KL_loss_function(mu,sigma)

        if self.training:
            eps = (torch.randn_like(mu)).cuda()
            extracted_feature = mu + torch.mul(eps,sigma)
        else:
            extracted_feature = mu

        return extracted_feature, KL_loss

    def forward(self, x):

        extracted_feature, KL_loss = self.IB_feature_extraction(x)

        output = self.classifier(extracted_feature)

        return output, KL_loss


class MVCNN_DIB(Model):

    def __init__(self, name, model, nclasses=40, num_views=12, hid_dim1 = 16, hid_dim2 = 24 ,bits = 1):
        super(MVCNN_DIB, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.bits = bits
        #self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        #self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()


        self.net_1 = model.net_1

        net_2_list = []

        for _ in range(12):
            net_2_list.append(copy.deepcopy(model.net_2))

        self.net_2 = nn.ModuleList(net_2_list)


        IB_mu_list = []
        IB_sigma_list = []

        for _ in range(12):
            IB_mu_list.append(copy.deepcopy(model.IB_mu))

        for _ in range(12):
            IB_sigma_list.append(copy.deepcopy(model.IB_sigma))

        self.IB_mu = nn.ModuleList(IB_mu_list)
        self.IB_sigma = nn.ModuleList(IB_sigma_list)


        encoder1 = []
        for _ in range(12):
            encoder1.append(
                            nn.Sequential(
                            nn.Linear(40,256),
                            nn.ReLU(),
                            nn.Linear(256,256),
                            nn.ReLU(),
                            nn.Linear(256,self.hid_dim1),
                            nn.Sigmoid()
                            ))
        self.encoder1 = nn.ModuleList(encoder1)


        encoder2 = []
        for _ in range(12):
            encoder2.append(
                            nn.Sequential(
                            nn.Linear(40,256),
                            nn.ReLU(),
                            nn.Linear(256,256),
                            nn.ReLU(),
                            nn.Linear(256,self.hid_dim2),
                            nn.Sigmoid()
                            ))
        self.encoder2 = nn.ModuleList(encoder2)


        Linear = nn.Linear(40,self.num_views)
        Linear.bias.data = Linear.bias.data + 1

        self.attention_module = nn.Sequential(
                                nn.Linear(self.hid_dim1*self.num_views,40),
                                nn.ReLU(),
                                Linear,
                                #nn.Linear(40,self.num_views),
                                nn.Sigmoid()
                                )

        self.feature1_classifier = nn.Sequential(
                                nn.Linear((self.hid_dim1)*self.num_views,40),
                                nn.ReLU(),
                                nn.Linear(40,40)
                                )

        self.final_classifier = nn.Sequential(
                                nn.Linear((self.hid_dim1+self.hid_dim2)*self.num_views,40),
                                nn.ReLU(),
                                nn.Linear(40,40)
                                )

        single_view_classifier = []
        for _ in range(12):
            single_view_classifier.append(
                                nn.Sequential(
                                nn.Linear((self.hid_dim1+self.hid_dim2),40),
                                nn.ReLU(),
                                nn.Linear(40,40)
                                ))

        self.single_view_classifier = nn.ModuleList(single_view_classifier)




    def forward(self, x):


        extracted_feature = self.net_1(x)
        extracted_feature = extracted_feature.view(extracted_feature.shape[0],-1)

        extracted_feature = extracted_feature.reshape((-1, 12, 25088)) # (batch, view , dim)

        encoder1_output = []
        encoder2_output = []

        for i in range(12):
            single_view_feature = extracted_feature[:,i]
            single_view_feature = self.net_2[i](single_view_feature)
            single_view_mu = 10 * self.IB_mu[i](single_view_feature)
            single_view_sigma = self.IB_sigma[i](single_view_feature)

            if self.training:
                eps = (torch.randn_like(single_view_mu)).cuda()
                single_view_feature = single_view_mu + torch.mul(eps,single_view_sigma)
            else:
                single_view_feature = single_view_mu

            single_view_feature1 = self.encoder1[i](single_view_feature)
            single_view_feature2 = self.encoder2[i](single_view_feature)

            encoder1_output.append(single_view_feature1)
            encoder2_output.append(single_view_feature2)        

        feature1 = torch.cat(encoder1_output, dim = 1)
        feature1 = feature1.reshape((-1, self.hid_dim1))  # (batch * view, hid_dim1)

        feature2 = torch.cat(encoder2_output, dim = 1)
        feature2 = feature2.reshape((-1, self.hid_dim2))  # (batch * view, hid_dim2)

        quantized_feature1 = torch.round(feature1)
        quantized_feature1 = (quantized_feature1 - feature1).detach() + feature1 # (batch * view, hid_dim1)
        quantized_feature1 = quantized_feature1 - 0.5

        quantized_feature2 = torch.round(feature2)
        quantized_feature2 = (quantized_feature2 - feature2).detach() + feature2 # (batch * view, hid_dim2)
        quantized_feature2 = quantized_feature2 - 0.5 # {-0.5,0.5}

        quantized_feature_concat = torch.cat((quantized_feature1,quantized_feature2),dim=1) # (batch * view, hid_dim1 + hid_dim2)

        attention_score = self.attention_module(quantized_feature1.reshape(-1,self.num_views * self.hid_dim1)) # (batch, view)

        binary_attention_score = torch.round(attention_score)

        binary_attention_score = (binary_attention_score - attention_score).detach() + attention_score

        received_feature2 = torch.mul(binary_attention_score.unsqueeze(dim = -1),quantized_feature2.reshape((-1,self.num_views,self.hid_dim2)) )

        received_feature2 = received_feature2.reshape((-1,self.hid_dim2)) # (batch * view, hid_dim2)

        views_output = []
        quantized_feature_concat = quantized_feature_concat.reshape((-1,self.num_views,self.hid_dim1 + self.hid_dim2))

        for i in range(12):
            single_view_feature = quantized_feature_concat[:,i]
            single_view_output = self.single_view_classifier[i](single_view_feature)
            views_output.append(single_view_output)

        views_output = torch.cat(views_output, dim = 1)
        views_output = views_output.reshape((-1, 40))  # (batch * view, 40)

        feature1_output = self.feature1_classifier(quantized_feature1.reshape((-1,(self.hid_dim1)*self.num_views))) # (batch, 40)
        received_feature_concat = torch.cat((quantized_feature1,received_feature2),dim=1) # (batch * view, hid_dim1 + hid_dim2)

        final_output = self.final_classifier(received_feature_concat.reshape((-1,(self.hid_dim1+self.hid_dim2)*self.num_views))) # (batch, 40)

        return final_output, feature1_output, views_output, binary_attention_score

