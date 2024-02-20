# Traffic Behavior and Interaction Simulation Setup

This repository provides instructions for setting up a simulation environment tailored for analyzing traffic behavior and interactions. It builds upon foundational work by NVlabs, specifically leveraging the following repositories:

- [Traffic Behavior Simulation](https://github.com/NVlabs/traffic-behavior-simulation/tree/main)
- [Trajdata](https://github.com/NVlabs/trajdata)

The setup outlined here facilitates the installation of necessary tools and libraries for conducting traffic simulations and analyzing trajectory data.

## Environment Setup

Follow these steps to create and activate a new Conda environment:

1. **Create the Conda Environment**:
    ```sh
    conda create -n trajdata_interaction python=3.9
    ```

    This command creates a new environment named `trajdata_interaction` with Python 3.9.

2. **Activate the Environment**:
    ```sh
    conda activate trajdata_interaction
    ```

    Activating the environment configures your terminal session to use the Python version and libraries installed in `trajdata_interaction`.

## Package Installation

With the environment set up and activated, proceed to install the `trajdata` package and its dependencies:

1. **Install `trajdata`**:
    ```sh
    pip install trajdata
    ```

    This command installs the base `trajdata` package.

2. **Install the `interaction` Module**:
    ```sh
    pip install "trajdata[interaction]"
    ```

    Including the `interaction` module adds functionalities tailored for interaction-aware traffic simulation.
3. ```sh
    pip install -e .
    ```


Training Command
python scripts/train.py --dataset_path <Interactiondataset-path>  --output_dir test --checkpoint=test
