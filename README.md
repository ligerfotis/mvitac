
# MViTac: Multimodal Visual-Tactile Representation Learning through Self-Supervised Contrastive Pre-Training

The repository was jointly developed with [Vedant Dave](https://github.com/VDML97)

[Project Website](https://sites.google.com/view/mvitac/home)

This repository contains the implementation for training and evaluating the MVitAC model. The model is trained using contrastive learning on the Calandra dataset. The code was developed and tested on **Ubuntu 22.04** with **Python 3.10**.
## Overview

1. **mvitac.ipynb**: This notebook contains the procedures for pretraining the MViTac model using contrastive learning. It covers configuration setup, model architecture, training, and evaluation.
2. **evaluate.ipynb**: A notebook dedicated to evaluating pretrained models. It provides metrics and visual results to understand the model's performance.
3. **contrastive_learning_dataset.py**: A script that prepares datasets specifically for contrastive learning.
4. **generate_dataset.py**: This script generates datasets, likely preprocessing and structuring data in a way suitable for training and evaluation.
5. **model.py**: Defines the architecture of the MVitAC model.
6. **utils.py**: Contains utility functions that assist in training, evaluation, and other tasks.
7. **config.py**: Contains configuration settings and parameters that dictate various aspects of training and evaluation.

## How to Use

1. **Setup**: Ensure all dependencies are installed. This can typically be done using a `requirements.txt` file.
2. **Data Preparation**: - Download the dataset from [this link](https://drive.google.com/drive/folders/1wHEg_RR8YAQjMnt9r5biUwo5z3P6bjR3) and place it in the `mvitac/calandradataset/` folder. Use the `generate_dataset.py` script to prepare your data. This will preprocess and structure your data into a suitable format.
3. **Training**: Open the `mvitac.ipynb` notebook and follow the steps to pretrain the MViTac model.
4. **Evaluation**: Once the model is trained, use the `evaluate.ipynb` notebook to train a linear classifier on the top of the learned representations to evaluate its performance on test data.
5. **Utilities**: The `utils.py` script contains helper functions. It's integrated into the training and evaluation notebooks, so there's no need to run it separately.

## How to use - Docker

As an alternative to the above steps, we used Docker to implement a straight-forward way to get this project running. The easiest way for a new user is the [Devcontainer Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in the [Visual Studio Code IDE](https://code.visualstudio.com/). Hence, when you open this repository in Visual Studio code, it will ask you to open the workspace inside a container. Click yes.

Also you might want to build the Docker image yourself and run it manually. Follow these steps:

1. [Install](https://docs.docker.com/get-docker/) Docker obviously
2. Clone this repository and cd into it
3. Build the image: `docker build .devcontainer -t mvitac`
4. Run the container: `docker run -it -p 8888:8888 -v ${PWD}:/home/mvitacdev/mvitacdev_ws mvitac bash`

### Additional Notes - GPU
You might want to [use your GPU inside of the container](https://github.com/NVIDIA/nvidia-container-toolkit).
This part will be updated.

## Notes

- Ensure the data paths in the configuration are correctly set.
- Monitor the training progress to ensure convergence. Adjust hyperparameters in `config.py` if necessary.
- For custom datasets, ensure they are structured correctly and update data paths in the configuration.

# Cite
    @misc{dave2024multimodal,
      title={Multimodal Visual-Tactile Representation Learning through Self-Supervised Contrastive Pre-Training}, 
      author={Vedant Dave and Fotios Lygerakis and Elmar Rueckert},
      year={2024},
      eprint={2401.12024},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
    }

## License

MIT License

Copyright (c) [2023] [Chair of Cyber-Physical Systems]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
