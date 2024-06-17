import os
import yaml
import time
import pandas as pd
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as datasets

from data.datasets import BackboneDataset
from data.datasets import CWDataset
from models.ResNet50 import res50


def main():
    # Get config params. See config.yaml for parameters
    # This is the only global variable
    global config
    conf_path = "config.yaml"
    config = get_config(conf_path)

    # Set the random seeds for reproducibility
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    cudnn.benchmark = True

    # Creating the Model
    # TODO: Make the model have 200 (or 199) output neurons instead of whatever it is now
    model = res50(pretrained_model=config["dir"]["model"])


    # Defining Loss and Optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # TODO: Other optimizers?
    optimizer = torch.optim.SGD(model.parameters(), lr=config["optim"]["lr"],
                           momentum=config["optim"]["momentum"],
                           weight_decay=config["optim"]["l2"])
    scheduler = StepLR(optimizer, step_size=config["optim"]["lr_step"], gamma=config["optim"]["lr_gamma"])

    # Distribute batches across all GPUs with DataParallel                       
    model = torch.nn.DataParallel(model, device_ids=list(range(config["ngpu"])))
    model = model.cuda()

    # ============
    # Data Loading
    # ============
    # Getting Data Directories
    train_df = pd.read_parquet(os.path.join(config["dir"]["data"], "train.parquet"))
    traindir = os.path.join(config["dir"]["data"], "train")
    # TODO: Validation split
    test_df = pd.read_parquet(os.path.join(config["dir"]["data"], "test.parquet"))
    testdir = os.path.join(config["dir"]["data"], "test")

    # Tell PIL not to skip truncated images, just try its best to get the whole thing
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # TODO: path exists in annotations, so we dont need imagedir?
    train_loader = DataLoader(
        BackboneDataset(image_folder=traindir, annotations=train_df, 
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=True,
        num_workers=4)

    test_loader = DataLoader(
        BackboneDataset(image_folder=testdir, annotations=test_df, 
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=False,
        num_workers=4)
    
    # Concept Loaders are the folders containing instances of a particular concept, across all classes
    # TODO: Add support for learned concepts. This can probably be done by adding a row to 
    # the parquet with the same high level concept, but an "unk_k" token for low level
    # concepts, where the k is the id of the particularl unknown concept
    concept_loaders = []
    

    # =============
    # Training Loop
    # =============
    if config["verbose"]:
        print("Starting Training")
    
    best_acc = 0
    accs = []
    for epoch in range(config["train"]["epochs"]):
        # Train and validate an epoch
        train_acc, train_dur = train(train_loader, concept_loaders, model, criterion, optimizer, epoch)
        # TODO: add a concept trainer here if the epoch is a multiple of 10 or something?
        accs.append(validate(val_loader, model, criterion, epoch))

        # Learning Rate Scheduler Step. Every 30 epochs, lr /= 5
        scheduler.step()
        
        # Compute avg acc of last 5 epochs
        num_epochs = len(accs)
        last_n_accuracies = accs[-min(num_epochs, 5):]
        avg_acc = sum(last_n_accuracies) / len(last_n_accuracies)

        # Only save if the average accuracy is better
        is_best = avg_acc > best_acc
        if is_best:
            best_acc = avg_acc
            # We'll need to reload the best model, so save the path to its checkpoint
            best_path = save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'acc': avg_acc
                        })

        if (config["verbose"]) and ((epoch + 1) % config["train"]["print_freq"] == 0):
            print(f"Epoch {epoch + 1} Complete")
            print(f"\tDuration: {train_dur:.2f}s\n\tTrain Accuracy: {train_acc:.4f}")
            print(f"Val Accuracy: {accs[-1]:.4f}, Was Best? {is_best}")

    # Load the best model before validating
    model.load_model(best_path)
    val_acc = validate(test_loader, model, criterion, epoch)
    if config["verbose"]:
        print(f"Training Completed. Final Accuracy: {val_acc:.4f}")


def train(train_loader, concept_loaders, model, criterion, optimizer):
    """
    Trains the model for one epoch
    """
    start = time.time()
    total_loss = 0
    model.train()
    for i, (input, target) in enumerate(train_loader):
        # Every 30 epochs, update CW layers?
        # TODO: See if this is right. i is the batch index, NOT epoch
        if (i + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                # Update the gradient matrix G for the CW layers
                for concept_index, concept_loader in enumerate(concept_loaders):
                    # Setting the mode tells the model what concept should be activated right now
                    # i.e. Mode 0: pointy beak, 1: curved beak, ...
                    model.change_mode(concept_index)
                    for batch, _ in concept_loader:
                        batch = batch.cuda()
                        model(batch)
                        # Only do one batch -- why??
                        break
                model.update_rotation_matrix()
                # Stop computing the gradient for CW w/ mode=-1
                # -1 is the default mode that skips gradient computation
                model.change_mode(-1)
            model.train()
        
        # Move them to CUDA, assumes CUDA access
        target = target.cuda()
        input = input.cuda()
        
        # Forward pass + loss computation
        output = model(input)
        loss = criterion(output, target)
        total_loss += loss.item()
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    end = time.time()
    # TODO: replace this with accuracy
    avg_loss = total_loss / len(train_loader)

    return avg_loss, end - start
  

def validate(val_loader, model, criterion):
    """
    Validation forward passes for the current model
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input, target in val_loader:
            # Move input and target to GPUs
            target = target.cuda()
            input = input.cuda()
            
            # Forward Pass
            output = model(input)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    # TODO: make this accuracy
    avg_loss = total_loss / len(val_loader)
    return avg_loss


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


def save_checkpoint(state):
    """
    Save the model in a compatible format with the ResNet50 load_model method

    Params:
    -------
    - state (dict): dictionary with keys [epoch, acc, state_dict]

    Returns:
    --------
    - path (str): Path to file
    """
    path = os.path.join(config["dir"]["checkpoint"], 
                        f"{config["dir"]["cp_prefix"]}_epoch{state["epoch"]}_acc{state["acc"]}.pth")
    torch.save(state["state_dict"], path)
    return path


# This function computes the accuracy of the top k classes
# i.e. maybe pred != target, but the second highest prediction was the target
# So you're if you had top2 acc, this would be marked as a correct 
def accuracy(output, target, topk=(1,)):
    # TODO: make this work
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