from data.datasets import BackboneDataset, CWDataset
from models.ResNet50 import ResNet, res50

import os
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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR


def main():
    global CONFIG
    with open("config.yaml", 'r') as file:
        CONFIG = yaml.safe_load(file)

    # ============
    # Data Loading
    # ============
    # Get data directories
    cub_path = os.getenv('CUB_PATH')
    train_df = pd.read_parquet(os.path.join(cub_path, CONFIG["directories"]["data"], "train.parquet"))
    test_df = pd.read_parquet(os.path.join(cub_path, CONFIG["directories"]["data"], "test.parquet"))
    # Ensure reproducibility
    train_df, val_df = train_test_split(train_df, test_size=len(test_df), random_state=CONFIG["seed"])

    # Tell PIL not to skip truncated images, just try its best to get the whole thing
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Load the low and high level concept dictionaries
    low_path = os.path.abspath(CONFIG["directories"]["low_concepts"])
    high_path = os.path.abspath(CONFIG["directories"]["high_concepts"])
    mappings_path = os.path.abspath(CONFIG["directories"]["mappings"])
    with open(low_path, "r") as file:
        low_level = json.load(file)
    with open(high_path, "r") as file:
        high_level = json.load(file)
    with open(mappings_path, "r") as file:
        mappings = json.load(file)

    # Make the backbone and concept loaders. We only need a concept loader for the train set.
    # Train
    train_loader = DataLoader(
        BackboneDataset(
            annotations=train_df,
            transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=CONFIG["train"]["batch_size"],
        shuffle=True,
        num_workers=CONFIG["train"]["workers"]
    )

    # Validation
    val_loader = DataLoader(
        BackboneDataset(
            annotations=val_df,
            transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=CONFIG["train"]["batch_size"],
        shuffle=False,
        num_workers=CONFIG["train"]["workers"]
    )

    # Test
    test_loader = DataLoader(
        BackboneDataset(
            annotations=test_df,
            transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=CONFIG["train"]["batch_size"],
        shuffle=False,
        num_workers=CONFIG["train"]["workers"]
    )

    # Concept
    # First, we need to add the free concepts using CWDataset's static method
    train_df_free, free_low_level, free_mappings = CWDataset.make_free_concepts(train_df, 2, low_level,
                                                                                high_level, mappings)
    high_to_low = CWDataset.generate_high_to_low_mapping(free_low_level, free_mappings)
    min_concept = min(free_low_level.values())
    max_concept = max(free_low_level.values())

    concept_loaders = []
    for i in range(min_concept, max_concept + 1):
        concepts = CWDataset(
            train_df_free, low_level, mode=i,
            transform=transforms.Compose([transforms.ToTensor()])
        )

        # Sometimes concepts are not in the training set, so we temporarily skip them.
        # This can be better addressed in the future.
        if len(concepts) == 0:
            continue

        # Notice that num workers is not set. If there are hundreds of concepts,
        # using lots of workers per concept is not practical.
        concept_loader = DataLoader(
            concepts,
            batch_size=CONFIG["train"]["batch_size"],
            shuffle=True
        )
        concept_loaders.append(concept_loader)

    # ==============
    # Model Creation
    # ==============
    # Set the random seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    cudnn.benchmark = True

    # Create the model. If you are using the default backbone, make sure to set vanilla pretrain to True.
    model = res50(
        whitened_layers=CONFIG["cw_layer"]["whitened_layers"],
        high_to_low=high_to_low,
        cw_lambda=CONFIG["cw_layer"]["cw_lambda"],
        activation_mode=CONFIG["cw_layer"]["activation_mode"],
        pretrained_model=CONFIG["directories"]["model"],
        vanilla_pretrain=CONFIG['train']['vanilla']
    )

    # Define loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG["optim"]["lr"],
        momentum=CONFIG["optim"]["momentum"],
        weight_decay=CONFIG["optim"]["l2"])
    scheduler = StepLR(optimizer, step_size=CONFIG["optim"]["lr_step"], gamma=CONFIG["optim"]["lr_gamma"])

    # Distribute batches across all GPUs with DataParallel
    model = torch.nn.DataParallel(model, device_ids=list(range(CONFIG["ngpu"])))
    model = model.cuda()

    # =========================
    # Print Training Parameters
    # =========================
    print(
        "Training parameters:\n"
        f"{yaml.dump(CONFIG['cw_layer'], default_flow_style=False)}\n\n"
        f"{yaml.dump(CONFIG['train'], default_flow_style=False)}\n\n"
        f"{yaml.dump(CONFIG['optim'], default_flow_style=False)}\n\n"
    )

    # =============
    # Training Loop
    # =============
    if CONFIG["verbose"]:
        print("Starting training...")

    best_acc = 0
    accs = []
    for epoch in range(CONFIG["train"]["epochs"]):
        # Train and validate an epoch
        _, train_acc, train_duration = train(train_loader, concept_loaders, model, criterion, optimizer)
        accs.append(validate(val_loader, model, criterion)[1])

        # Learning rate scheduler step. Every 30 epochs, learning rate lr is divided by 5.
        scheduler.step()

        # Compute average accuracy of last 5 epochs
        num_epochs = len(accs)
        last_n_accuracies = accs[-min(num_epochs, 5):]
        avg_acc = sum(last_n_accuracies) / len(last_n_accuracies)

        # Only save if the average accuracy is better
        is_best = (avg_acc > best_acc)
        if is_best:
            best_acc = avg_acc
            # We'll need to reload the best model, so save the path to its checkpoint
            best_path = save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict(), "acc": avg_acc})

        if (CONFIG["verbose"]) and ((epoch + 1) % CONFIG["train"]["print_freq"] == 0):
            print(
                f"Epoch {epoch + 1} complete\n"
                f"\tDuration: {train_duration:.2f}s\n"
                f"\tTrain Accuracy: {train_acc:.4f}\n"
                f"\tValidation Accuracy: {accs[-1]:.4f}, was best? {is_best}",
                flush=True
            )

    # Load the best model before validating
    model.module.load_model(best_path)
    _, val_acc = validate(test_loader, model, criterion)
    if CONFIG["verbose"]:
        print(f"Training completed. Final Accuracy: {val_acc:.4f}")


def train(
    train_loader: DataLoader,
    concept_loaders: list[DataLoader],
    model: nn.DataParallel[ResNet],
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer
):
    """
    Trains the model for one epoch.
    """
    start = time.time()
    total_loss = 0
    total_correct = 0
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input: torch.Tensor
        target: torch.Tensor
        # NOTE: CUB dataset labels start at 1, hence this line. If your target starts at zero, this needs to be removed!
        target = target - 1

        # Train for concept whitening loss once every 30 batches.
        if (i + 1) % CONFIG["train"]["train_cw_freq"] == 0:
            model.eval()
            with torch.no_grad():
                # Update the gradient matrix G for the concept whitening layers.
                # Each concept in the CWLayer is indexed by its corresponding position in concept_loaders.
                for idx, concept_loader in enumerate(concept_loaders):
                    model.module.change_mode(idx)

                    for batch, region in concept_loader:
                        batch: torch.Tensor
                        batch = batch.cuda()
                        model(batch, region, batch.shape[2])  # batch.shape[2] gives the original x dimension
                        break  # only sample one batch for each concept

                model.module.update_rotation_matrix()
                # Stop computing the gradient for concept whitening.
                # mode=-1 is the default mode that skips gradient computation.
                model.module.change_mode(-1)
            model.train()

        # Move them to CUDA, assumes CUDA access
        input = input.cuda()
        target = target.cuda()

        # Forward pass + loss computation
        output = model(input)
        loss: torch.Tensor = criterion(output, target)

        # Performance metrics
        total_loss += loss.item()
        total_correct += top_k_correct(output, target)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.time()
    avg_loss = total_loss / len(train_loader)
    acc = total_correct / len(train_loader.dataset)

    return avg_loss, acc, end - start


def validate(
    data_loader: DataLoader,
    model: nn.DataParallel[ResNet],
    criterion: nn.modules.loss._Loss
):
    """
    Gradient-free forward pass for the current model.
    """
    total_loss = 0
    total_correct = 0
    model.eval()

    with torch.no_grad():
        for input, target in data_loader:
            input: torch.Tensor
            target: torch.Tensor
            # NOTE: Cub datasets labels start at 1, hence this line. If your target starts at zero,
            # this needs to be removed!
            target = target - 1

            # Moves them to CUDA, assumes CUDA access
            target = target.cuda()
            input = input.cuda()

            # Forward pass
            output = model(input)
            loss: torch.Tensor = criterion(output, target)

            # Performance metrics
            total_loss += loss.item()
            total_correct += top_k_correct(output, target)

    avg_loss = total_loss / len(data_loader)
    acc = total_correct / len(data_loader.dataset)

    return avg_loss, acc


def save_checkpoint(state):
    """
    Save the model in a compatible format with the ResNet50 `load_model` method.

    Params:
    - state (dict): Dictionary with keys [epoch, acc, state_dict]. Contains the model's state information.

    Returns:
        str: Path to the saved model file.
    """
    path = os.path.join(CONFIG["directories"]["checkpoint"],
                        f"{CONFIG['train']['checkpoint_prefix']}_epoch_{state['epoch']}_acc_{state['acc']:.4f}.pth")
    torch.save(state["state_dict"], path)
    return path


def top_k_correct(output, target: torch.Tensor, k=1):
    """
    See how many predictions were in the top k predicted classes.
    Assumes target is already adjusted to be -1 for CUB.
    """
    _, predicted_topk = torch.topk(output, k, dim=1)
    correct_topk = (predicted_topk == target.unsqueeze(1)).sum().item()
    return correct_topk


if __name__ == '__main__':
    main()
