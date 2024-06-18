import os
import sys
import json
import yaml
import time
import pandas as pd
from PIL import ImageFile
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

sys.path.insert(1, "/home/users/aak61/CS474/ConceptWhitening")
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
    model = res50(pretrained_model=config["dirs"]["model"])

    print(type(config['optim']['lr']))


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
    train_df = pd.read_parquet(os.path.join(config["dirs"]["data"], "train.parquet"))
    test_df = pd.read_parquet(os.path.join(config["dirs"]["data"], "test.parquet"))
    train_df, val_df = train_test_split(train_df, test_size=len(test_df), random_state=config["seed"])  # Ensuring reproducibility

    # Tell PIL not to skip truncated images, just try its best to get the whole thing
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Load the low and high level concept dictionaries
    low_path = os.path.abspath(config["dirs"]["low_dict"])
    high_path = os.path.abspath(config["dirs"]["high_dict"])
    with open(low_path, "r") as file:
        low_level = json.load(file)
    with open(high_path, "r") as file:
        high_level = json.load(file)

    # Make the backbone and concept loaders
    # Only need a concept loader for the train set
    # Train
    train_loader = DataLoader(
        BackboneDataset(annotations=train_df, 
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=True,
        num_workers=4
    )

    # Validation
    val_loader = DataLoader(
        BackboneDataset(annotations=val_df, 
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=False,
        num_workers=4
    )

    # Test
    test_loader = DataLoader(
        BackboneDataset(annotations=test_df, 
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
        batch_size=config["train"]["batch_size"], 
        shuffle=False,
        num_workers=4
    )

    # Concept
    concepts = CWDataset(train_df, high_level, low_level, n_free=0, 
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ]))
    concept_loader = DataLoader(
        concepts,
        batch_size=config["train"]["batch_size"], 
        shuffle=True,
        num_workers=4
    )
    

    # =============
    # Training Loop
    # =============
    if config["verbose"]:
        print("Starting Training")
    
    best_acc = 0
    accs = []
    for epoch in range(config["train"]["epochs"]):
        # Train and validate an epoch
        train_loss, train_acc, train_dur = train(train_loader, concept_loader, concepts, model, criterion, optimizer)
        # TODO: add a concept trainer here if the epoch is a multiple of 10 or something?
        accs.append(validate(val_loader, model, criterion)[1])

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
            print(f"Epoch {epoch + 1} Complete", flush=True)
            print(f"\tDuration: {train_dur:.2f}s\n\tTrain Accuracy: {train_acc:.4f}", flush=True)
            print(f"Val Accuracy: {accs[-1]:.4f}, Was Best? {is_best}", flush=True)

    # Load the best model before validating
    model.load_model(best_path)
    val_acc = validate(test_loader, model, criterion, epoch)
    if config["verbose"]:
        print(f"Training Completed. Final Accuracy: {val_acc:.4f}")


def train(train_loader, concept_loader, concept_dataset, model, criterion, optimizer):
    """
    Trains the model for one epoch
    """
    start = time.time()
    total_loss = 0
    total_correct = 0
    model.train()
    for i, (input, target) in enumerate(train_loader):
        # Every 30 epochs, update CW layers?
        # TODO: See if this is right. i is the batch index, NOT epoch
        # BUG: This is important, target is offset by 1 in CUB
        target = target - 1 
        if (i + 1) % 1000000000 == 0:
            model.eval()
            with torch.no_grad():
                # Update the gradient matrix G for the CW layers
                for i in range(1, concept_dataset.n_concepts + 1):
                    concept_dataset.set_mode(i)
                    model.change_mode(i)
                    for batch, region in concept_loader:
                        batch = batch.cuda()
                        # batch.shape[2] gives the original x dimension
                        model(batch, region, batch.shape[2])
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

        # Performance Metrics
        total_loss += loss.item()
        total_correct += correct(output, target)
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    end = time.time()
    avg_loss = total_loss / len(train_loader)
    acc = total_correct / len(train_loader.dataset)

    return avg_loss, acc, end - start
  

def validate(dataloader, model, criterion):
    """
    Grad free forward passes for the current model
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for input, target in dataloader:
            # BUG: This is important, target is offset by 1 in CUB
            target = target - 1 
            # Move input and target to GPUs
            target = target.cuda()
            input = input.cuda()
            
            # Forward Pass
            output = model(input)
            loss = criterion(output, target)

            # Performance Metrics
            total_loss += loss.item()
            total_correct += correct(output, target, k=1)
    
    avg_loss = total_loss / len(dataloader)
    acc = total_correct / len(dataloader.dataset)
    return avg_loss, acc


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
    path = os.path.join(config["dirs"]["checkpoint"], 
                        f"{config['dirs']['cp_prefix']}_epoch{state['epoch']}_acc{state['acc']}.pth")
    torch.save(state["state_dict"], path)
    return path


def correct(output, target, k=1):
    """
    See how many predictions were in the top k predicted classes.
    Assumes target is already adjusted to be -1 for CUB
    """
    _, predicted_topk = torch.topk(output, k, dim=1)
    correct_topk = (predicted_topk == target.unsqueeze(1)).sum().item()
    return correct_topk


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


if __name__ == '__main__':
    main()