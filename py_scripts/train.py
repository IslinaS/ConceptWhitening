import os
import sys
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile, Image

# Alex Imports
import yaml
from models.ResNet50 import res50


def get_config(path):
    """
    Opens the config file

    Params:
    -------
    - path (str): File path to load from

    Returns:
    --------
    - config (dict): Config parameters set in config.yaml
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # TODO: Do we need this checkpoints folder?
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    # Get config params. See config.yaml for parameters
    conf_path = "config.yaml"
    config = get_config(conf_path)

    # Set the random seeds for reproducibility
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    random.seed(config["seed"])

    # Creating the Model
    model = res50(pretrain=True,
                  pretrained_model=config["dir"]["model"])


    # Defining Loss and Optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["optim"]["lr"],
                           momentum=config["optim"]["momentum"],
                           weight_decay=config["optim"]["l2"]) # Option to use other optimizers?

    # Distribute batches across all GPUs with DataParallel                       
    model = torch.nn.DataParallel(model, device_ids=list(range(config["ngpu"])))
    model = model.cuda()

    """WHAT IS THIS?"""
    cudnn.benchmark = True

    # Getting Data Directories
    traindir = os.path.join(config["dir"]["data"], "train")
    valdir = os.path.join(config["dir"]["data"], "val")
    testdir = os.path.join(config["dir"]["data"], "test")
    conceptdir_train = os.path.join(config["dir"]["data"], "concept_train")
    conceptdir_test = os.path.join(config["dir"]["data"], "concept_test")

    # Normalize transform, see if the backbone uses this...
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # Tell PIL not to skip truncated images, just try its best to get the whole thing
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=True)

    """WHAT IS args.concepts?"""
    # Concept Loaders are the folders containing instances of a particular concept, across all classes
    concept_loaders = [
        torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(conceptdir_train, concept), transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=True)
        for concept in args.concepts.split(',')
    ]

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=False)


    print("Start training")
    best_acc = 0
    accs = []
    for epoch in range(config["train"]["epochs"]):
        # TODO: LR decay?
        # Train an epoch, store the validation accuracy
        train(train_loader, concept_loaders, model, criterion, optimizer, epoch)
        accs.append(validate(val_loader, model, criterion, epoch))
        
        # Compute avg acc of last 5 epochs
        num_epochs = len(accs)
        last_n_accuracies = accs[-min(num_epochs, 5):]
        avg_acc = sum(last_n_accuracies) / len(last_n_accuracies)

        # Only save if the average accuracy is better
        is_best = avg_acc > best_acc
        if is_best:
            best_acc = avg_acc
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': avg_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, config["dirs"]["cp_prefix"])

    validate(test_loader, model, criterion, epoch)


def train(train_loader, concept_loaders, model, criterion, optimizer, epoch):
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Every 30 epochs, update CW layers
        if (i + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                # Update the gradient matrix G for the CW layers
                for concept_index, concept_loader in enumerate(concept_loaders):
                    # Setting the mode tells the model what concept should be activated right now
                    # i.e. Mode 0: pointy beak, 1: curved beak, ...
                    model.module.change_mode(concept_index)
                    for batch, _ in concept_loader:
                        X_var = torch.autograd.Variable(batch).cuda()
                        model(X_var)
                        break
                model.module.update_rotation_matrix()
                # change to ordinary mode
                model.module.change_mode(-1)
            model.train()
        
        
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
  

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
    return top1.avg


def plot_figures(args, model, test_loader_with_path, train_loader, concept_loaders, conceptdir):
    concept_name = args.concepts.split(',')

    if not os.path.exists('./plot/'+'_'.join(concept_name)):
        os.mkdir('./plot/'+'_'.join(concept_name))
    
    if args.evaluate == 'plot_top50':
        print("Plot top50 activated images")
        model = load_resnet_model(args, arch = 'resnet_cw', depth=18, whitened_layer='8')
        plot_concept_top50(args, test_loader_with_path, model, '8', activation_mode = args.act_mode)
    elif args.evaluate == 'plot_auc':
        print("Plot AUC-concept_purity")
        print("Note: this requires multiple models trained with CW on different layers")
        aucs_cw = plot_auc_cw(args, conceptdir, '1,2,3,4,5,6,7,8', plot_cpt = concept_name, activation_mode = args.act_mode)
        print("Running AUCs svm")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_svm = plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir, '1,2,3,4,5,6,7,8', plot_cpt = concept_name, model_type = 'svm')
        print("Running AUCs lr")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_lr = plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir, '1,2,3,4,5,6,7,8', plot_cpt = concept_name, model_type = 'lr')
        print("Running AUCs best filter")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_filter = plot_auc_filter(args, model, conceptdir, '1,2,3,4,5,6,7,8', plot_cpt = concept_name)
        print("AUC plotting")
        plot_auc(args, 0, 0, 0, 0, plot_cpt = concept_name)
        print("End plotting")
    
def save_checkpoint(state, is_best, prefix, checkpoint_folder='./checkpoints'):
    if args.arch == "resnet_cw" or args.arch == "densenet_cw" or args.arch == "vgg16_cw":
        # save checkpoints for model with CW layer
        concept_name = '_'.join(args.concepts.split(','))
        if not os.path.exists(os.path.join(checkpoint_folder,concept_name)):
            os.mkdir(os.path.join(checkpoint_folder,concept_name))
        filename = os.path.join(checkpoint_folder,concept_name,'%s_checkpoint.pth.tar'%prefix)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_folder,concept_name,'%s_model_best.pth.tar'%prefix))
    elif args.arch == "resnet_original" or args.arch == "densenet_original" or args.arch == "vgg16_original":
        # save checkpoints for model without CW layer
        filename = os.path.join(checkpoint_folder,'%s_checkpoint.pth.tar'%prefix)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_folder,'%s_model_best.pth.tar'%prefix))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()