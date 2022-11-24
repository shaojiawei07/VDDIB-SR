import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
#from tensorboardX import SummaryWriter
import time
import copy
import os
import datetime

class SingleViewTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, num_views=12, args = None):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.num_views = num_views
        self.datetime = datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S_')
        self.args = args
        self.model.cuda()


    def train_IB(self, n_epochs):

        best_acc = 0
        
        for epoch in range(n_epochs):
            self.model.train()
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            for i, data in enumerate(self.train_loader):

                in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data, KL_loss = self.model(in_data)

                # IB objective function
                loss = self.loss_fn(out_data, target) + self.args.gamma * KL_loss 
                
                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]

                loss.backward()
                self.optimizer.step()
                
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch+1, i+1, loss, acc)
                if (i+1)%30==0:
                    print(log_str)

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    val_overall_acc = self.update_validation_accuracy(epoch)
                    print("test accuracy",val_overall_acc)

            # save the model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                torch.save(self.model.state_dict(),"./saved_model/phase1_pretrained_model_" + self.datetime +".pth")
            print("best accuracy:",best_acc)
 
            # adjust learning rate
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5


    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        with torch.no_grad():

            for _, data in enumerate(self.val_loader, 0):

                in_data = Variable(data[1]).cuda()
                target = Variable(data[0]).cuda()

                out_data, _ = self.model(in_data)
                pred = torch.max(out_data, 1)[1]
                results = pred == target

                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]

        print ('Total # of test models: ', all_points)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()

        print ('test acc. : ', val_overall_acc)

        self.model.train()

        return val_overall_acc 

class MultipleViewTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 num_views, args):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.num_views = num_views
        self.args = args
        self.datetime = datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S_')

        self.model.cuda()
        self.best_acc = 0

    def train_DIB(self, n_epochs):

        saved_model = []

        for param in self.model.net_1.parameters():
            param.requires_grad = False
        for param in self.model.net_2.parameters():
            param.requires_grad = False
        for param in self.model.IB_mu.parameters():
            param.requires_grad = False
        for param in self.model.IB_sigma.parameters():
            param.requires_grad = False

        for epoch in range(n_epochs):
            self.model.train()

            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            for i, data in enumerate(self.train_loader):

                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda() # unpack
                target = Variable(data[0]).cuda().long() # unpack

                self.optimizer.zero_grad()

                final_output, internal_output, aux_output, attention_score = self.model(in_data)

                average_active_views = torch.mean(torch.norm(attention_score,p=1,dim=1))

                active_views_loss = torch.mean(F.relu(torch.norm(attention_score,p=1,dim=1)-self.args.view_number_constraint))

                broad_cast_target = target.unsqueeze(dim = -1) # (batch, 1)
                sample_tensor = (torch.zeros((final_output.size()[0],self.model.num_views))).cuda() # (batch, view)
                broad_cast_target, _ = torch.broadcast_tensors(broad_cast_target, sample_tensor) # (batch, view)
                broad_cast_target = broad_cast_target.reshape((-1))

                loss = 0.5 * self.loss_fn(final_output, target) + 0.5 * self.loss_fn(internal_output, target)  \
                    + self.args.beta * self.loss_fn(aux_output, broad_cast_target) + self.args.beta * active_views_loss

                pred = torch.max(final_output, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]

                loss.backward()
                self.optimizer.step()
                
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f; avg_active_views %.3f' % (epoch+1, i+1, loss, acc, average_active_views.item())
                if (i+1)%30==0:
                    print(log_str)

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    val_overall_acc = self.update_validation_accuracy(epoch)


            # save best model
            if val_overall_acc > self.best_acc:
                self.best_acc = val_overall_acc
                torch.save(self.model.state_dict(),"./saved_model/phase2_model" + "_dim1_" + str(self.model.hid_dim1) + "_dim2_" + str(self.model.hid_dim2) + self.datetime +".pth")
            print("best accuracy:",self.best_acc, "\n")

        for param in self.model.net_2.parameters():
            param.requires_grad = True
        for param in self.model.IB_mu.parameters():
            param.requires_grad = True
        for param in self.model.IB_sigma.parameters():
            param.requires_grad = True

    def fine_tune(self, n_epochs):

        print("\nModel fine tuning ...")

        for param in self.model.net_1.parameters():
            param.requires_grad = False
        for param in self.model.net_2.parameters():
            param.requires_grad = True
        for param in self.model.IB_mu.parameters():
            param.requires_grad = True
        for param in self.model.IB_sigma.parameters():
            param.requires_grad = True

        self.train_DIB(n_epochs)


    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        self.model.eval()

        with torch.no_grad():

            for _, data in enumerate(self.val_loader, 0):

                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                target = Variable(data[0]).cuda()

                out_data , _, _  ,_ = self.model(in_data)
                pred = torch.max(out_data, 1)[1]
                results = pred == target

                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]

        print ('Total # of test models: ', all_points)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        print ('val overall acc. : ', val_overall_acc)
        self.model.train()

        return val_overall_acc

    def cascade_inference(self,retransmission_threshold):
        all_correct_points = 0
        all_points = 0

        out1_correct_points = 0
        out2_correct_points = 0

        all_loss = 0
        count_one_trans = 0
        count_view = 0

        view_retrans_count = torch.zeros(12).cuda()

        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):

                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                target = Variable(data[0]).cuda()

                final_output , internal_output, _, attention_score = self.model(in_data)

                out_data1 = internal_output
                out_data2 = final_output

                coefficient, _ = torch.max(F.softmax(out_data1,dim=1),dim=1, keepdim=True)
                coefficient = torch.where((coefficient)>(retransmission_threshold),torch.ones(1).to('cuda'),torch.zeros(1).to('cuda'))

                retrans_attention_score = torch.mul((1-coefficient),attention_score)
                view_retrans_count += torch.sum(retrans_attention_score,dim = 0)

                count_one_trans += torch.norm(coefficient, p=1)
                output_cascade = torch.mul(out_data1,coefficient) + torch.mul(out_data2,(1-coefficient))
                pred_cascade = output_cascade.argmax(dim=1, keepdim=False)

                results = pred_cascade == target

                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]


                pred_output1 = out_data1.argmax(dim=1, keepdim=False)
                pred_output2 = out_data2.argmax(dim=1, keepdim=False)
                output1_results = pred_output1 == target
                output2_results = pred_output2 == target
                correct_points1 = torch.sum(output1_results.long())
                correct_points2 = torch.sum(output2_results.long())
                out1_correct_points += correct_points1
                out2_correct_points += correct_points2


            acc = all_correct_points.float() / all_points
            cascade_acc = acc.cpu().data.numpy()

            avg_view_in_retrans = torch.sum(view_retrans_count) / (all_points - count_one_trans)


            communication_overhead = self.args.hid_dim1 * self.args.bits * self.args.num_views \
                                     + self.args.hid_dim2 * self.args.bits * avg_view_in_retrans * (1 - count_one_trans.float()/all_points)

            print("Threshold:",retransmission_threshold,"Accuracy: {:.2f} %,".format(cascade_acc * 100) ,"Communication overhead: {:.1f} bits".format(communication_overhead.item()))


