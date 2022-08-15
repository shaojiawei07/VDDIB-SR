from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import copy

#import matplotlib.pyplot as plt
import numpy as np



parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--dim1', type=int, default=3)
parser.add_argument('--dim2', type=int, default=3)
parser.add_argument('--bit',type = int, default = 1)
parser.add_argument('--beta',type = float, default = 1e-5)
parser.add_argument('--seed',type = int, default = 1)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args

        self.view1_feature_extraction = nn.Sequential(
                              nn.Linear(392,256),
                              nn.ReLU(),
                              nn.Linear(256,256),
                              nn.ReLU()
                            )
        self.view2_feature_extraction = nn.Sequential(
                              nn.Linear(392,256),
                              nn.ReLU(),
                              nn.Linear(256,256),
                              nn.ReLU()
                            )

        self.view1_ib_mu = nn.Sequential(
                              nn.Linear(256,32),
                              nn.Tanh()
                            )
        self.view2_ib_mu = nn.Sequential(
                              nn.Linear(256,32),
                              nn.Tanh()
                            )

        self.view1_ib_sigma = nn.Sequential(
                              nn.Linear(256,32),
                              nn.Sigmoid()
                            )
        self.view2_ib_sigma = nn.Sequential(
                              nn.Linear(256,32),
                              nn.Sigmoid()
                            )

        self.view1_encoder1 = nn.Sequential(
                                nn.Linear(32,args.dim1),
                                nn.Sigmoid()
                            )

        self.view1_encoder2 = nn.Sequential(
                                nn.Linear(32,args.dim2),
                                nn.Sigmoid()
                            )

        self.view2_encoder1 = nn.Sequential(
                                nn.Linear(32,args.dim1),
                                nn.Sigmoid()
                            )
        self.view2_encoder2 = nn.Sequential(
                                nn.Linear(32,args.dim2),
                                nn.Sigmoid()
                            )

        '''

        self.attention_module = nn.Sequential(
                                nn.Linear(args.dim1 * 2,8),
                                nn.ReLU(),
                                nn.Linear(8,2)
                            )
        '''

        self.view_specific_decoder1 = nn.Sequential(
                        nn.Linear(args.dim1+args.dim2, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                        )
        self.view_specific_decoder2 = nn.Sequential(
                        nn.Linear(args.dim1+args.dim2, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                        )

        self.decoder1 = nn.Sequential(
                        nn.Linear(2 * (args.dim1), 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        )

        self.decoder2 = nn.Sequential(
                        nn.Linear(2 * (args.dim1+args.dim2), 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        )

    def forward(self, x, args):

        feature1, feature2, KL_loss = self.IB_feature_extraction(x)

        encoded_view1_1 = self.view1_encoder1(feature1)
        encoded_view2_1 = self.view2_encoder1(feature2)

        encoded_view1_2 = self.view1_encoder2(feature1)
        encoded_view2_2 = self.view2_encoder2(feature2)

        quantized_view1_1 = (torch.round(encoded_view1_1) - encoded_view1_1).detach() + encoded_view1_1 - 0.5
        quantized_view2_1 = (torch.round(encoded_view2_1) - encoded_view2_1).detach() + encoded_view2_1 - 0.5

        quantized_view1_2 = (torch.round(encoded_view1_2) - encoded_view1_2).detach() + encoded_view1_2 - 0.5
        quantized_view2_2 = (torch.round(encoded_view2_2) - encoded_view2_2).detach() + encoded_view2_2 - 0.5

        view_specific_feature1 = torch.cat((quantized_view1_1,quantized_view1_2),dim=1)
        view_specific_feature2 = torch.cat((quantized_view2_1,quantized_view2_2),dim=1)

        received_feature_T1 = torch.cat((quantized_view1_1,quantized_view2_1),dim=1)
        received_feature_T2 = torch.cat((view_specific_feature1,view_specific_feature2),dim=1)

        view_specific_output1 = self.view_specific_decoder1(view_specific_feature1)
        view_specific_output2 = self.view_specific_decoder2(view_specific_feature2)

        T1_output = self.decoder1(received_feature_T1)
        T2_output = self.decoder2(received_feature_T2)

        return F.log_softmax(T2_output, dim=1), F.log_softmax(T1_output, dim=1), \
               F.log_softmax(view_specific_output1, dim=1), F.log_softmax(view_specific_output2, dim=1), \
               KL_loss

    def IB_feature_extraction(self,x):

        view1 = x[:,:,0:14,:]
        view2 = x[:,:,14:,:]

        view1 = torch.reshape(view1,(-1,392))
        view2 = torch.reshape(view2,(-1,392))

        feature_view1 = self.view1_feature_extraction(view1)
        feature_view2 = self.view2_feature_extraction(view2)

        mu1 = 10 * self.view1_ib_mu(feature_view1) 
        sigma1 = self.view1_ib_sigma(feature_view1) 

        mu2 = 10 * self.view2_ib_mu(feature_view2) 
        sigma2 = self.view2_ib_sigma(feature_view2)

        KL_loss = self.KL_loss(mu1,sigma1) + self.KL_loss(mu2,sigma2)

        if self.training:
            eps1 = (torch.randn_like(mu1)).to(device)
            feature1 = mu1 + torch.mul(eps1,sigma1)
            eps2 = (torch.randn_like(mu2)).to(device)
            feature2 = mu2 + torch.mul(eps2,sigma2)
        else:
            feature1 = mu1
            feature2 = mu2

        return feature1, feature2, KL_loss

    def KL_loss(self,mu1,sigma1,sigma2 = 1):


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

    def IB_extractor_requires_grad(self, requires_grad = False):

        for params in self.view1_feature_extraction.parameters():
            params.requires_grad =  requires_grad

        for params in self.view2_feature_extraction.parameters():
            params.requires_grad =  requires_grad

        for params in self.view1_ib_mu.parameters():
            params.requires_grad =  requires_grad

        for params in self.view2_ib_mu.parameters():
            params.requires_grad =  requires_grad

        for params in self.view1_ib_sigma.parameters():
            params.requires_grad =  requires_grad

        for params in self.view2_ib_sigma.parameters():
            params.requires_grad =  requires_grad      


def train_VDDIB_SR(args, model, device, train_loader, optimizer, epoch):

    model.IB_extractor_requires_grad(requires_grad = False)
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        T2_output, T1_output, view1_output, view2_output, _ = model(data, args)
        loss = F.nll_loss(T2_output, target) + F.nll_loss(T1_output, target) \
        + args.beta * (((F.nll_loss(view1_output, target) + F.nll_loss(view2_output, target)))) #+ args.ga * KL_loss
        loss.backward()
        optimizer.step()
    return loss

def train_VIB(args, model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, _, view1_output, view2_output, KL_loss = model(data, args)
        loss = F.nll_loss(view1_output, target) + F.nll_loss(view2_output, target) + 1e-4 * KL_loss

        loss.backward()
        optimizer.step()
    return loss

def fine_tune(args, model, device, train_loader, optimizer, epoch):

    model.IB_extractor_requires_grad(requires_grad = True)
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        T2_output, T1_output, view1_output, view2_output, KL_loss = model(data, args)
        loss = F.nll_loss(T2_output, target) + F.nll_loss(T1_output, target) \
        + args.beta * (((F.nll_loss(view1_output, target) + F.nll_loss(view2_output, target)))) #+ 1e-4 * KL_loss
        loss.backward()
        optimizer.step() 



def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    correct_internal = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            T2_output,T1_output,_,_,_ = model(data, args)
            test_loss += F.nll_loss(T2_output, target, reduction='sum').item()

            pred = T2_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            pred_internal = T1_output.argmax(dim=1, keepdim=True)
            correct_internal += pred_internal.eq(target.view_as(pred_internal)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * (correct) / len(test_loader.dataset) , 100. * ( correct_internal) / len(test_loader.dataset)


def inference(args, model, device, test_loader, threshold):
    model.eval()

    count_one_trans = 0
    all_points = 0
    all_correct_points = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            T2_output,T1_output,_,_,_ = model(data, args)

            coefficient,_ = torch.max(F.softmax(T1_output,dim=1),dim=1, keepdim=True)
            coefficient = torch.where((coefficient)>(threshold),torch.ones(1).to('cuda'),torch.zeros(1).to('cuda'))

            count_one_trans += torch.norm(coefficient, p=1)
            output_cascade = torch.mul(T1_output,coefficient) + torch.mul(T2_output,(1-coefficient))
            pred_cascade = output_cascade.argmax(dim=1, keepdim=False)
            results = pred_cascade == target
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        acc = all_correct_points.float() / all_points
        acc = acc.cpu().data.numpy()

        retrainsmission_ratio =   1 - (count_one_trans.float()/all_points).item()
        communication_cost = 2 * (args.dim1 * args.bit + args.dim2 * args.bit * retrainsmission_ratio)

        return acc, communication_cost


def main():
    

    kwargs = {'num_workers': 4, 'pin_memory': True} #if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(args).to(device)

    best_acc2 = 0
    #test_acc = 0
    #save_mask_dim = 0
    best_model = []

    optimizer1 = optim.SGD(model.parameters(), lr=args.lr)
    scheduler1 = StepLR(optimizer1, step_size=10, gamma=args.gamma)
    for epoch in range(0,args.epochs): # pretrain the feature extractor(s)
        loss = train_VIB(args, model, device, train_loader, optimizer1, epoch)
        print('VIB training ... epoch:', epoch, "loss", loss.item())
        scheduler1.step()

    optimizer2 = optim.SGD(model.parameters(), lr=args.lr)
    scheduler2 = StepLR(optimizer2, step_size=10, gamma=args.gamma)
    for epoch in range(0,args.epochs): # train the DVIB
        loss = train_VDDIB_SR(args, model, device, train_loader, optimizer2, epoch)
        print('VDDIB-SR training ... epoch:', epoch, "loss", loss.item())
        scheduler2.step()
        acc1, acc2 = test(args, model, device, test_loader)

        if acc2 >best_acc2:
            best_acc2 = acc2
            #test_acc = acc1
            best_model = copy.deepcopy(model.state_dict())

    
    optimizer3 = optim.SGD(model.parameters(), lr=args.lr )
    scheduler3 = StepLR(optimizer3, step_size=10, gamma=args.gamma)

    for epoch in range(0,args.epochs):
        print('Fine-tuning ... epoch:',epoch)
        fine_tune(args, model, device, train_loader, optimizer3, epoch)
        scheduler3.step()
        
        acc1, acc2 = test(args, model, device, test_loader)
        if acc2 >best_acc2:
            best_acc2 = acc2
            #test_acc = acc1
            best_model = copy.deepcopy(model.state_dict())
        

    

    # inference
    model.load_state_dict(best_model)
    threshold_list = [0.8,0.825,0.85,0.875,0.9,0.925,0.95]

    print("\nInference: \n")
    for i in range(len(threshold_list)):
        accuracy, cost  = inference(args, model, device, test_loader, threshold = threshold_list[i])
        print("threshold {:.3f}".format(threshold_list[i]),"accuracy {:.4f} %".format(accuracy * 100), "communication cost {:.2f} bits".format(cost))


if __name__ == '__main__':
    seed_torch(args.seed)
    main()

