import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange

from data.util_bmnist import get_dataset, IdxDataset
from models.pnd import PnDNet
from util import GeneralizedCELoss, EMA


class Learner(object):
    def __init__(self, args):
        self.model = args.model
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.temperature = args.temperature
        print(f'model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {args.exp}...')
        self.log_dir = os.makedirs(os.path.join(args.log_dir, args.dataset, args.exp), exist_ok=True)
        self.device = torch.device(args.device)
        self.args = args

        print(self.args)
        print(f'batch size: {self.batch_size}...')

        # logging directories
        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.result_dir, exist_ok=True)

        self.train_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=args.percent,
            use_preprocess=None,
        )
        self.valid_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="valid",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=None,
        )

        self.test_dataset = get_dataset(
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="test",
            transform_split="test",
            percent=args.percent,
            use_preprocess=None,
        )

        train_target_attr = []
        for data in self.train_dataset.data:     
            train_target_attr.append(int(data.split('/')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)
        print(args.data_dir, train_target_attr.shape, torch.max(train_target_attr))

        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]
        self.train_dataset = IdxDataset(self.train_dataset)
        self.train_target_attr = train_target_attr


        self.sample_loss_ema_b0 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device = self.device)
        self.sample_loss_ema_d0 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device = self.device)
        self.sample_loss_ema_b1 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device = self.device)
        self.sample_loss_ema_d1 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device = self.device)
        self.sample_loss_ema_b2 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device = self.device)
        self.sample_loss_ema_d2 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device = self.device)
        self.sample_loss_ema_b3 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device = self.device)
        self.sample_loss_ema_d3 = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=args.ema_alpha, device = self.device)


        # make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = nn.CrossEntropyLoss(reduction='none')

        print(f'self.criterion: {self.criterion}')
        print(f'self.bias_criterion: {self.bias_criterion}')


        self.best_valid_acc_b  = 0
        self.best_valid_acc = 0
        print('finished model initialization....')
       

    def contrastive_loss(self, y_mix, y_ori, indices_mini):
        temperature = self.temperature
        y_pos, y_neg = y_mix
        y_pos = y_pos.unsqueeze(1)
        y_neg = y_neg.unsqueeze(1)
        bs = int(y_ori.size(0)/16)
        y_neg = torch.reshape(y_neg, (y_pos.size(0),bs, y_pos.size(2)))
        y_ori = y_ori[indices_mini]
        y_ori = y_ori.unsqueeze(1)
        y_all = torch.cat([y_pos, y_neg], dim=1)
        y_expand = y_ori.repeat(1, 9, 1)     
        neg_dist = -((y_expand - y_all) ** 2).mean((2)) * temperature
        label = torch.zeros(16).long().to(self.device)
        contrastive_loss_euclidean = nn.CrossEntropyLoss()(neg_dist, label)
        return contrastive_loss_euclidean

    def kl_loss(self, x_pred, x_gt):
        kl_gt = F.softmax(x_gt, dim=-1)
        kl_pred = F.log_softmax(x_pred, dim=-1)
        tmp_loss = F.kl_div(kl_pred, kl_gt, reduction='none')
        tmp_loss = torch.exp(-tmp_loss).mean()
        return tmp_loss


   # evaluation code for ours
    def evaluate_ours(self, data_loader):
        self.model.eval()
        total_correct, total_num = 0, 0
        total_correct_cls = np.zeros(10)
        total_num_cls = np.zeros(10)
        loss_all = []
        for data, attr, index in tqdm(data_loader, leave=False, desc= 'Evaluation'):
            label = attr[:, 0]
            data = data.to(self.device)
            label = label.to(self.device)     
            with torch.no_grad():
                _, logits_gate  = self.model(data)
                pred_label = logits_gate['dm_conflict_out']
                loss_i = self.intri_criterion(pred_label, label)
                loss_all.append(loss_i.mean().item())
                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]
                for i in range(10):
                    correct_i = (pred[label==i] == label[label==i]).long()
                    total_correct_cls[i] += correct_i.sum()
                    total_num_cls[i] += correct_i.shape[0]

        loss_all = np.mean(loss_all)
        accs = total_correct/float(total_num)
        accs_cls = np.divide(total_correct_cls,total_num_cls)
        return accs,accs_cls,loss_all

    def save_ours(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, f"best_model.th")
        else:
            model_path = os.path.join(self.result_dir, "model_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')


    def board_ours_acc(self, step):
        # check label network
        valid_accs, valid_accs_cls, valid_loss = self.evaluate_ours(data_loader = self.valid_loader) 
        if valid_accs >= self.best_valid_acc:
            self.best_valid_acc = valid_accs
            self.save_ours(step, best=True)
        print(f'valid acc: {valid_accs}')
        print(f'valid loss : {valid_loss}')
        print(f'valid accs cls: {valid_accs_cls}')
        print(f'BEST valid acc: {self.best_valid_acc}')

        test_accs, test_accs_cls, test_loss = self.evaluate_ours(data_loader = self.test_loader) 
        print(f'test acc: {test_accs}')
        print(f'test loss: {test_loss}')
        print(f'test accs cls: {test_accs_cls}')
        

    def final_board_ours_acc(self):
        valid_accs, valid_accs_cls, valid_loss = self.evaluate_ours(data_loader = self.valid_loader) 
        print(f'Final valid acc: {valid_accs}')
        print(f'Final valid loss: {valid_loss}')
        print(f'Final valid accs cls: {valid_accs_cls}')

        test_accs, test_accs_cls, test_loss = self.evaluate_ours(data_loader = self.test_loader) 
        print(f'Final test acc: {test_accs}')
        print(f'Final test loss: {test_loss}')
        print(f'Final test accs cls: {test_accs_cls}')


    def train_ours_step1(self, args):
        print('************** Step1 Training Starts... ************** ')
        self.model = PnDNet(self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr1,
            weight_decay=args.weight_decay,
        )
        self.scheduler1 = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_decay_step1, gamma=args.lr_gamma)

        self.intri_criterion = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(q=args.loss_q)
        print(f'criterion: {self.criterion}')
        print(f'intri criterion: {self.intri_criterion}')
        print(f'bias criterion: {self.bias_criterion}')
        
        train_iterator = trange(int(args.num_epochs1), desc="Epoch")
        for epoch in train_iterator:
            self.model.train() 
            print(f"self.optimizer1 lr: { self.optimizer.param_groups[-1]['lr']}")
            for batch in tqdm(self.train_loader, desc='Iteration'):
                self.optimizer.zero_grad()   
                index, data, attr, image_path = batch
                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx].to(self.device)   
                logits_all, logits_gate = self.model(data, y = label, use_mix = False)  
                ######experts
                loss_dis_conflict_all = torch.zeros((4,data.size(0))).float()
                loss_dis_align_all = torch.zeros((4,data.size(0))).float()         
                for i in range(4):
                    key_conflict_i = f"E={i}, dm_conflict_out"
                    key_align_i = f"E={i}, dm_align_out"   
                    pred_conflict_i = logits_all[key_conflict_i]
                    pred_align_i = logits_all[key_align_i]    
                    loss_dis_conflict_i_ = self.intri_criterion(pred_conflict_i, label).detach()
                    loss_dis_align_i_ = self.intri_criterion(pred_align_i, label).detach()

                    # EMA sample loss
                    getattr(self, f'sample_loss_ema_d{i}').update(loss_dis_conflict_i_, index)
                    getattr(self, f'sample_loss_ema_b{i}').update(loss_dis_align_i_, index)

                    # class-wise normalize
                    loss_dis_conflict_i_ = getattr(self, f'sample_loss_ema_d{i}').parameter[index].clone().detach()
                    loss_dis_align_i_ = getattr(self, f'sample_loss_ema_b{i}').parameter[index].clone().detach()

                    loss_dis_conflict_i_ = loss_dis_conflict_i_.to(self.device)
                    loss_dis_align_i_ = loss_dis_align_i_.to(self.device)

                    for c in range(self.num_classes):
                        class_index = torch.where(label == c)[0].to(self.device)
                        max_loss_conflict = getattr(self, f'sample_loss_ema_d{i}').max_loss(c)
                        max_loss_align = getattr(self, f'sample_loss_ema_b{i}').max_loss(c)
                        loss_dis_conflict_i_[class_index] /= max_loss_conflict
                        loss_dis_align_i_[class_index] /= max_loss_align

                    loss_weight_i  = loss_dis_align_i_ / (loss_dis_align_i_+ loss_dis_conflict_i_ + 1e-8)     
                    loss_dis_conflict_i = self.intri_criterion(pred_conflict_i, label) * loss_weight_i.to(self.device) 
                    loss_dis_align_i = self.bias_criterion(pred_align_i, label)                                            
                    loss_dis_conflict_all[i,:] = loss_dis_conflict_i
                    loss_dis_align_all[i,:] = loss_dis_align_i                
                loss_dis_experts  = args.alpha1*loss_dis_conflict_all.mean() +  loss_dis_align_all.mean()
                
                kl_loss_conflict_1 = self.kl_loss(logits_all['E=1, dm_align_out'], logits_all['E=0, dm_align_out'].detach())
                kl_loss_conflict_2 = self.kl_loss(logits_all['E=2, dm_align_out'], logits_all['E=1, dm_align_out'].detach())         
                kl_loss_conflict_3 = self.kl_loss(logits_all['E=3, dm_align_out'], logits_all['E=2, dm_align_out'].detach())  
                kl_loss_conflict = kl_loss_conflict_1 + kl_loss_conflict_2 + kl_loss_conflict_3
                            
                loss_experts = 4*loss_dis_experts + kl_loss_conflict     
           
                ######gate
                pred_conflict = logits_gate['dm_conflict_out']
                loss_dis_conflict = self.intri_criterion(pred_conflict, label)          
                loss_gate  = loss_dis_conflict.mean()                

                loss = loss_experts + loss_gate           
                loss.backward()
                self.optimizer.step()   
         
            self.scheduler1.step()
                    

            if epoch % args.lr_decay_step1 == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer1 lr: { self.optimizer.param_groups[-1]['lr']}")

            self.board_ours_acc(epoch)           
            print(f'finished epoch: {epoch}')
  
  
    def train_ours_step2(self, args):
        print('************** Step2 Training Starts... ************** ')

        self.model = PnDNet(self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.result_dir, f'best_model.th'))['state_dict'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr2,
            weight_decay=args.weight_decay,
        )
        self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_decay_step2, gamma=args.lr_gamma)

        self.intri_criterion = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(q=args.loss_q)
        print(f'criterion: {self.criterion}')
        print(f'intri criterion: {self.intri_criterion}')
        print(f'bias criterion: {self.bias_criterion}')
        
        train_iterator = trange(int(args.num_epochs2), desc="Epoch")
        
        for epoch in train_iterator:
            self.model.train()
            print(f"self.optimizer2 lr: { self.optimizer.param_groups[-1]['lr']}")
            for batch in tqdm(self.train_loader, desc='Iteration'):
                self.optimizer.zero_grad()   
                index, data, attr, image_path = batch
                data = data.to(self.device)
                attr = attr.to(self.device)         
                label = attr[:, args.target_attr_idx].to(self.device)
                logits_all, logits_gate = self.model(data, y = label, use_mix = True)               
                ######experts
                loss_dis_conflict_all = torch.zeros((4,data.size(0))).float()
                loss_dis_align_all = torch.zeros((4,data.size(0))).float()
                loss_cf = torch.zeros((4,16)).float()
                for i in range(4):
                    key_conflict_i = f"E={i}, dm_conflict_out"
                    key_align_i = f"E={i}, dm_align_out"   
                    pred_conflict_i = logits_all[key_conflict_i]
                    pred_align_i = logits_all[key_align_i]    
                    loss_dis_conflict_i_ = self.intri_criterion(pred_conflict_i, label).detach()
                    loss_dis_align_i_ = self.intri_criterion(pred_align_i, label).detach()

                    # EMA sample loss
                    getattr(self, f'sample_loss_ema_d{i}').update(loss_dis_conflict_i_, index)
                    getattr(self, f'sample_loss_ema_b{i}').update(loss_dis_align_i_, index)

                    # class-wise normalize
                    loss_dis_conflict_i_ = getattr(self, f'sample_loss_ema_d{i}').parameter[index].clone().detach()
                    loss_dis_align_i_ = getattr(self, f'sample_loss_ema_b{i}').parameter[index].clone().detach()

                    loss_dis_conflict_i_ = loss_dis_conflict_i_.to(self.device)
                    loss_dis_align_i_ = loss_dis_align_i_.to(self.device)

                    for c in range(self.num_classes):
                        class_index = torch.where(label == c)[0].to(self.device)
                        max_loss_conflict = getattr(self, f'sample_loss_ema_d{i}').max_loss(c)
                        max_loss_align = getattr(self, f'sample_loss_ema_b{i}').max_loss(c)
                        loss_dis_conflict_i_[class_index] /= max_loss_conflict
                        loss_dis_align_i_[class_index] /= max_loss_align

                    loss_weight_i  = loss_dis_align_i_ / (loss_dis_align_i_+ loss_dis_conflict_i_ + 1e-8)  
                    loss_dis_conflict_i = self.intri_criterion(pred_conflict_i, label) * loss_weight_i.to(self.device) 
                    loss_dis_align_i = self.bias_criterion(pred_align_i, label)
                    loss_dis_conflict_all[i,:] = loss_dis_conflict_i
                    loss_dis_align_all[i,:] = loss_dis_align_i                              
                    key_out_mix = f"E={i}, dm_out_mix" 
                    key_indices_mini_i = f"E={i}, indices_mini" 
                    indices_mini_i = logits_all[key_indices_mini_i] 
                    pred_out_mix = logits_all[key_out_mix] 
                    loss_cf_i = self.contrastive_loss(pred_out_mix, pred_conflict_i, indices_mini_i)                                               
                    loss_cf[i,:] = loss_cf_i
                    
                kl_loss_conflict_1 = self.kl_loss(logits_all['E=1, dm_align_out'], logits_all['E=0, dm_align_out'].detach())
                kl_loss_conflict_2 = self.kl_loss(logits_all['E=2, dm_align_out'], logits_all['E=1, dm_align_out'].detach())         
                kl_loss_conflict_3 = self.kl_loss(logits_all['E=3, dm_align_out'], logits_all['E=2, dm_align_out'].detach())  
                kl_loss_conflict = kl_loss_conflict_1 + kl_loss_conflict_2 + kl_loss_conflict_3
                            
                loss_dis_experts  = args.alpha2*loss_dis_conflict_all.mean() +  loss_dis_align_all.mean()
                loss_cf_experts = loss_cf.mean()
                loss_experts = 4*loss_dis_experts + args.beta * loss_cf_experts + kl_loss_conflict                                                    # Eq.4 Total objective
            
                ######gate 
                pred_conflict = logits_gate['dm_conflict_out']
                loss_dis_conflict = self.intri_criterion(pred_conflict, label)
                loss_gate  = loss_dis_conflict.mean()
                
                loss = loss_experts + loss_gate              
                loss.backward()
                self.optimizer.step()   
            
            self.scheduler2.step()
                    
            if epoch % args.lr_decay_step2 == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer2 lr: { self.optimizer.param_groups[-1]['lr']}")
            
            self.board_ours_acc(epoch+args.num_epochs1)
            print(f'finished epoch: {epoch+args.num_epochs1}')
 

    def test_ours(self):
        self.intri_criterion = nn.CrossEntropyLoss(reduction='none')
        self.model = PnDNet(self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.result_dir, 'best_model.th'))['state_dict'])
        self.final_board_ours_acc()

