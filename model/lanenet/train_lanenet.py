import torch
import torch.nn as nn
import numpy as np
import time
import copy
from model.lanenet.loss import DiscriminativeLoss
from tqdm import tqdm

def ComputeLoss(net_output, binary_label, instance_label):
    k_binary = 10    
    k_instance = 0.3
    k_dist = 1.0

    loss_fn = nn.CrossEntropyLoss()
    
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]

    return total_loss, binary_loss, instance_loss, out

def TrainModel(model, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()
    training_log = {'epoch':[], 'training_loss':[], 'val_loss':[]}
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model, training_log = Training(model, dataloaders["train"], device, optimizer, dataset_sizes["train"], training_log)
        model, training_log, best_model_wts = Validation(best_loss, model, dataloaders["val"], device, optimizer, dataset_sizes["val"], training_log, best_model_wts)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    training_log['training_loss'] = np.array(training_log['training_loss'])

    model.load_state_dict(best_model_wts)
    return model, training_log

def Training(model,dataloaders,device,optimizer,dataset_sizes,training_log):
    model.train()
    running_loss = 0.0
    running_loss_b = 0.0
    running_loss_i = 0.0
    progressbar = tqdm(range(len(dataloaders)))
    for inputs, binarys, instances in dataloaders:
        
        inputs = inputs.type(torch.FloatTensor).to(device)
        binarys = binarys.type(torch.LongTensor).to(device)
        instances = instances.type(torch.FloatTensor).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ComputeLoss(outputs, binarys, instances)
        loss[0].backward()
        optimizer.step()
        
        running_loss += loss[0].item() * inputs.size(0)
        running_loss_b += loss[1].item() * inputs.size(0)
        running_loss_i += loss[2].item() * inputs.size(0)
        progressbar.set_description("batch loss: {:.3f}".format(loss[0].item()))
        progressbar.update(1)
    progressbar.close()
    epoch_loss = running_loss / dataset_sizes
    binary_loss = running_loss_b / dataset_sizes
    instance_loss = running_loss_i / dataset_sizes
    print('Train Total Loss: {:.4f} Binary Loss: {:.4f} Instance Loss: {:.4f}'.format(epoch_loss, binary_loss, instance_loss))
    training_log['training_loss'].append(epoch_loss)
    return model,training_log

def Validation(best_loss,model,dataloaders,device,optimizer,dataset_sizes,training_log,best_model_wts):
    model.eval()
    running_loss = 0.0
    running_loss_b = 0.0
    running_loss_i = 0.0
    progressbar = tqdm(range(len(dataloaders)))
    for inputs, binarys, instances in dataloaders:
        inputs = inputs.type(torch.FloatTensor).to(device)
        binarys = binarys.type(torch.LongTensor).to(device)
        instances = instances.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(inputs)
            loss = ComputeLoss(outputs, binarys, instances)
        running_loss += loss[0].item() * inputs.size(0)
        running_loss_b += loss[1].item() * inputs.size(0)
        running_loss_i += loss[2].item() * inputs.size(0)
        progressbar.set_description("batch loss: {:.3f}".format(loss[0].item()))
        progressbar.update(1)
    progressbar.close()
    epoch_loss = running_loss / dataset_sizes
    binary_loss = running_loss_b / dataset_sizes
    instance_loss = running_loss_i / dataset_sizes
    print('Validation Total Loss: {:.4f} Binary Loss: {:.4f} Instance Loss: {:.4f}'.format(epoch_loss, binary_loss, instance_loss))
    training_log['val_loss'].append(epoch_loss)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    return model, training_log, best_model_wts