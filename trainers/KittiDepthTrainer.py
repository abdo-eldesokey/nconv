########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from trainers.trainer import Trainer # from CVLPyDL repo
import torch

import matplotlib.pyplot as plt
import os.path
from utils.AverageMeter import AverageMeter
from utils.saveTensorToImages import saveTensorToImages
from utils.error_metrics import MAE, RMSE, MRE, Deltas

err_metrics = ['MAE', 'RMSE', 'MRE', 'Delta1', 'Delta2', 'Delta3']

class KittiDepthTrainer(Trainer):
    def __init__(self, net, params, optimizer, objective, lr_scheduler, dataloaders, dataset_sizes,
             workspace_dir, sets=['train', 'val'], use_load_checkpoint = None):
        
        # Call the constructor of the parent class (trainer)
        super().__init__(net, optimizer, lr_scheduler, objective, use_gpu=params['use_gpu'], workspace_dir=workspace_dir)
          
        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.use_load_checkpoint = use_load_checkpoint
        
        self.params = params
        self.save_chkpt_each = params['save_chkpt_each']
        self.sets = sets
        self.save_images = params['save_out_imgs']
        self.load_rgb = params['load_rgb'] if 'load_rgb' in params else False 
        
        
        for s in self.sets: self.stats[s+'_loss'] = []
                
   
####### Training Function #######

    def train(self, max_epochs):
        print('#############################\n### Experiment Parameters ###\n#############################')
        for k, v in self.params.items(): print('{0:<22s} : {1:}'.format(k,v))
                                                
        # Load last save checkpoint
        if self.use_load_checkpoint != None:
            if self.use_load_checkpoint > 0:
                print('\n=> Loading checkpoint {} ...'.format(self.use_load_checkpoint), end=' ')
                if self.load_checkpoint(self.use_load_checkpoint):
                    print('Checkpoint was loaded successfully!\n')
            elif self.use_load_checkpoint == -1:
                print('=> Loading last checkpoint ...', end=' ')
                if self.load_checkpoint():
                    print('Checkpoint was loaded successfully!\n')

        for epoch in range(self.epoch, max_epochs+1): # range function returns max_epochs-1
            self.epoch = epoch
                                   
            # Decay Learning Rate 
            self.lr_scheduler.step() # LR decay
            
            print('\nTraining Epoch {}: (lr={}) '.format(epoch, self.optimizer.param_groups[0]['lr']), end=' ')

            
            # Train the epoch
            loss_meter = self.train_epoch()
            
            # Add the average loss for this epoch to stats
            for s in self.sets: self.stats[s+'_loss'].append(loss_meter[s].avg)
            
            # Save checkpoint
            if self.use_save_checkpoint and (self.epoch) % self.save_chkpt_each == 0:                    
                self.save_checkpoint()
                print('\n => Checkpoint was saved successfully!')
                        
            
        # Save the final model
        torch.save(self.net, self.workspace_dir + '/final_model.pth')        
        
        print("Training Finished.")
            
        return self.net

    def train_epoch(self):
        device = torch.device("cuda:"+str(self.params['gpu_id']) if torch.cuda.is_available() else "cpu")
        
        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()
          
        for s in self.sets:
            # Iterate over data.
            for data in self.dataloaders[s]:
                if self.load_rgb:
                    inputs_d, C, labels, item_idxs, inputs_rgb = data
                    inputs_d=inputs_d.to(device) ; C=C.to(device)
                    labels=labels.to(device) ; inputs_rgb= inputs_rgb.to(device)
                    outputs, cout = self.net(inputs_d, C, inputs_rgb)
                else:
                    inputs_d, C, labels, item_idxs = data
                    inputs_d=inputs_d.to(device) ; C=C.to(device)
                    labels=labels.to(device) 
                    outputs, cout = self.net(inputs_d, C)                
                    
                
                # Calculate loss for valid pixel in the ground truth
                loss = self.objective(outputs, labels, cout, self.epoch)
                              

                # backward + optimize only if in training phase
                if s == 'train':                    
                    loss.backward()   
                    self.optimizer.step()            
                
                self.optimizer.zero_grad()
    
                # statistics
                loss_meter[s].update(loss.item(), inputs_d.size(0))
            
            print('[{}] Loss: {:.8f}'.format(s,  loss_meter[s].avg), end=' ')

        return loss_meter

    
####### Evaluation Function #######

    def evaluate(self):
        print('< Evaluate mode ! >')
       
        # Load last save checkpoint
        if self.use_load_checkpoint != None:
            if self.use_load_checkpoint > 0:
                print('=> Loading checkpoint {} ...'.format(self.use_load_checkpoint), end=' ')
                if self.load_checkpoint(self.use_load_checkpoint):
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')
            elif self.use_load_checkpoint == -1:
                print('=> Loading last checkpoint ...', end=' ')
                if self.load_checkpoint():
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')
        
        
        self.net.train(False)
        
        # AverageMeters for Loss
        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()
        
        # AverageMeters for error metrics  
        err = {}
        for m in err_metrics: err[m] = AverageMeter()
        
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')

        with torch.no_grad():
            for s in self.sets:
                print('Evaluating on [{}] set, Epoch [{}] ! \n'.format(s, str(self.epoch-1)))
                # Iterate over data.
                for data in self.dataloaders[s]:
                    
                    if self.load_rgb:
                        inputs_d, C, labels, item_idxs, inputs_rgb = data
                        inputs_d=inputs_d.to(device) ; C=C.to(device)
                        labels=labels.to(device) ; inputs_rgb= inputs_rgb.to(device)
                        outputs, cout = self.net(inputs_d, C, inputs_rgb)
                    else:
                        inputs_d, C, labels, item_idxs = data
                        inputs_d=inputs_d.to(device) ; C=C.to(device)
                        labels=labels.to(device) 
                        outputs, cout = self.net(inputs_d, C)   
                                    
                    
                    # Calculate loss for valid pixel in the ground truth
                    loss = self.objective(outputs, labels, cout, self.epoch)
                                
                    # statistics
                    loss_meter[s].update(loss.item(), inputs_d.size(0))

                    
                    # Convert data to depth in meters before error metrics
                    #outputs[outputs==0] = -1
                    if not self.load_rgb: 
                            outputs[outputs==outputs[0,0,0,0]] = -1
                    labels[labels==0] = -1
                    if self.params['invert_depth']:        
                        outputs = 1 / outputs
                        labels = 1 / labels
                    outputs[outputs==-1] = 0
                    labels[labels==-1] = 0
                    outputs *= self.params['data_normalize_factor']/256
                    labels *= self.params['data_normalize_factor']/256
                    
                    
                    # Calculate error metrics 
                    for m in err_metrics:
                        if m.find('Delta') >= 0:
                            fn = globals()['Deltas']() 
                            error = fn(outputs, labels)
                            err['Delta1'].update(error[0], inputs_d.size(0))
                            err['Delta2'].update(error[1], inputs_d.size(0))
                            err['Delta3'].update(error[2], inputs_d.size(0))
                            break 
                        else:    
                            fn = globals()[m]() 
                            error = fn(outputs, labels)
                            err[m].update(error.item(), inputs_d.size(0))
                    
                    # Save output images (optional)
                    if self.save_images and s in ['selval', 'test']:
                        outputs = outputs.data

                        outputs *= 256
                        
                        saveTensorToImages(outputs , item_idxs, os.path.join(self.workspace_dir, s+'_output_'+'epoch_'+str(self.epoch-1)))
                        saveTensorToImages(cout * 255, item_idxs, os.path.join(self.workspace_dir, s+'_cert_'+'epoch_'+str(self.epoch-1)))
    
                print('Evaluation results on [{}]:\n============================='.format(s))
                print('[{}]: {:.8f}'.format('Loss',  loss_meter[s].avg))
                for m in err_metrics: print('[{}]: {:.8f}'.format(m,  err[m].avg))
                            
                
                # Save evaluation metric to text file 
                fname = 'error_' + s + '_epoch_' + str(self.epoch-1) + '.txt'
                with open(os.path.join(self.workspace_dir, fname), 'w') as text_file:
                    text_file.write('Evaluation results on [{}], Epoch [{}]:\n==========================================\n'.format(s, str(self.epoch-1)))
                    text_file.write('[{}]: {:.8f}\n'.format('Loss',  loss_meter[s].avg))
                    for m in err_metrics: text_file.write('[{}]: {:.8f}\n'.format(m,  err[m].avg))
                        

           
