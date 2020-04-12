import os

import time
import copy
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from mega_age_asian_datasets import MegaAgeAsianDatasets
from SSR_Net_model import SSRNet
from config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, optimizer, criterion, hyper_parameters):
    
    # [TODO] What is this?
    global lr_scheduler
    
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(hyper_parameters.num_epochs):
        print('Epoch {}/{}'.format(epoch, hyper_parameters.num_epochs))
        print('-' * 10)
        
        for phase in sorted(dataloaders.keys()):
            print('mode: {}'.format(phase))

            if phase == 'train':
                model.train()
                torch.set_grad_enabled(True)
            else: # eval or test
                model.eval()  # Set model to evaluate mode
                torch.set_grad_enabled(False)
            
            loss = 0.0
            corrects_3 = 0
            corrects_5 = 0
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(hyper_parameters.device)
                labels = labels.to(hyper_parameters.device).float()
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # track history if only in train
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                loss += loss.item() * inputs.size(0)
                corrects_3 += torch.sum(torch.abs(outputs - labels) < 3)  # CA 3
                corrects_5 += torch.sum(torch.abs(outputs - labels) < 5)  # CA 5
            
            epoch_loss = loss / len(dataloaders[phase].dataset)
            CA_3 = corrects_3.double() / len(dataloaders[phase].dataset)
            CA_5 = corrects_5.double() / len(dataloaders[phase].dataset)
            
            
            print('{} Loss: {:.4f} CA_3: {:.4f}, CA_5: {:.4f}'.format(
                phase,
                epoch_loss,
                CA_3,
                CA_5
                ))

            time_elapsed = time.time() - since
            print('Complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60,
                time_elapsed % 60
                ))
            
            # deep copy the model
            if phase == 'val' and CA_3 > best_acc:
                best_acc = CA_3
                best_model_weights = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(CA_3)
        
        lr_scheduler.step(epoch)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60,
        time_elapsed % 60
        ))
    print('Best val CA_3: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, val_acc_history


def load_labels(dir_base, file_name):
    path = '{}/list/{}'.format(dir_base, file_name)
    labels = open(path).readlines()
    return labels

if __name__ == "__main__":
    
    hyper_parameters = Config()
    
    model = SSRNet(image_size=hyper_parameters.input_size)
    model = model.to(hyper_parameters.device)

    data_base_path = '../data/megaage_asian'

    image_labels = load_labels(data_base_path, 'train_name.txt')
    age_labels = load_labels(data_base_path, 'train_age.txt')

    
    random.seed(2019)
    random.shuffle(image_labels)
    train_size = int(len(image_labels) * 0.9)
    train_image_labels= image_labels[:train_size]
    val_image_labels = image_labels[train_size:]
    

    train_data_path = '{}/train'.format(data_base_path)

    random.seed(2019)
    random.shuffle(age_labels)
    assert len(image_labels) == len(age_labels), 'size mismatch, image_labels: {}, age_labels: {}'.format(len(image_labels), len(age_labels))
    train_age_labels = age_labels[:train_size]
    val_age_labels = age_labels[train_size:]

    train_datasets = MegaAgeAsianDatasets(
        train_image_labels, 
        train_age_labels, 
        train_data_path, 
        mode="train",
        augment=hyper_parameters.augmented,
        )
    
    val_datasets = MegaAgeAsianDatasets(
        val_image_labels,
        val_age_labels,
        train_data_path,
        mode="train",
        augment=hyper_parameters.augmented,
        )

    
    train_loader = DataLoader(
        train_datasets,
        batch_size=hyper_parameters.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
        )
    
    val_loader = DataLoader(
        val_datasets,
        batch_size=hyper_parameters.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    test_image_labels = load_labels(data_base_path, 'test_name.txt')
    test_age_labels = load_labels(data_base_path, 'test_age.txt')
    test_data_path = '{}/test'.format(data_base_path)

    test_datasets = MegaAgeAsianDatasets(
        test_image_labels,
        test_age_labels,
        test_data_path,
        mode="train",
        augment=hyper_parameters.augmented,
        )

    test_loader = DataLoader(
        test_datasets,
        batch_size=hyper_parameters.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
        )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyper_parameters.learning_rate,
        weight_decay=hyper_parameters.weight_decay)

    criterion = nn.L1Loss()

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hyper_parameters.step_size,
        gamma=hyper_parameters.gamma
        )
    
    # Train and evaluate
    model, hist = train_model(
        model,
        dataloaders,
        optimizer,
        criterion,
        hyper_parameters,
        )
    
    save_model_path = '../trained_models'
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)

    torch.save(
        model_to_train.to('cpu').state_dict(), 
    '{}/model_Adam_L1Loss_LRDecay_weightDecay{}_batch{}_lr{}_epoch{}_pretrained+90_64x64.pth'.format(
            save_model_path,
            hyper_parameters.weight_decay,
            hyper_parameters.batch_size,
            hyper_parameters.learning_rate,
            hyper_parameters.num_epochs
            )
    )
