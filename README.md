# ConceptWhitening

Use concept whitening combined with the ResNet50 model to distinguish bird species using important features.

Team members: Alex Katopodis, Hung Le, Islina Shan
Mentor: Prof. Cynthia Rudin, Duke University

Related research: [Concept Whitening](https://github.com/zhiCHEN96/ConceptWhitening.git)

Follow the instructions below to set up the environment, prepare the dataset, and run the training script.

## Setup

The paths to this repository and the CUB dataset may contain sensitive information, hence they will be supplied to the scripts via a `secrets.txt` file. In the `slurm_scripts` directory, create a `secrets.txt` file in the exact same format as the provided example `sample_secrets.txt`.

### Create the virtual environment and install dependencies

It's recommended to use a virtual environment to install and manage the many Python dependencies in this project.

1. In this directory, run

    ```bash
    python3 -m venv .venv
    pip3 install -r requirements.txt
    ```
2. If you have not already done so, set the variable   `VENV_PATH` in `secrets.txt` to be the **absolute** path to your virtual environment.

### Prepare the CUB dataset for training

1. Download and extract the CUB dataset [here](https://www.vision.caltech.edu/datasets/cub_200_2011/).
2. Set the variable `CUB_PATH` in `secrets.txt` to be the **absolute** path to your downloaded dataset.
3. In this directory, run
    ```bash
    ./slurm_scripts/make_data.sh
    ```
    or, if you are using slurm,
    ```bash
    sbatch slurm_scripts/make_data.sh
    ```
> [!NOTE]
> Remember to make this script executable by running
> ```bash
> sudo chmod +x slurm_scripts/make_data.sh
> ```

## Training the Model

To train the model, run the following command
```bash
./slurm_scripts/train.sh
```
or, if you are using slurm,
```bash
sbatch slurm_scripts/train.sh
```
> [!NOTE]
> Remember to make this script executable by running
> ```bash
> sudo chmod +x slurm_scripts/make_data.sh
> ```

> [!NOTE]
> Ensure you have enough disk space and memory to handle the dataset and model training.
> The training script will save model checkpoints and logs in the `checkpoints/` and `logs/` directories, respectively.

You can adjust all the training parameters in `config.yaml` as needed.

