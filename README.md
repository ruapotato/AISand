# AISand

This project implements a Convolutional Neural Network (CNN) model for predicting the next frame in a falling sand simulation game. It uses PyTorch to create and train a neural network that can learn the physics of particle interactions in a 2D grid.

## Features

- Custom dataset loader for falling sand simulation frames
- CNN-based model architecture
- Training loop with cross-entropy loss and learning rate scheduling
- Inference function for predicting the next frame
- Visualization of training progress and results
- Interactive script to run the trained model

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/falling-sand-cnn.git
   cd falling-sand-cnn
   ```

2. Install the required packages:
   ```
   pip install torch torchvision numpy matplotlib pillow tqdm
   ```

## Usage

### Training the Model

1. Prepare your dataset:
   Place your falling sand simulation frames in a directory structure as expected by the `SandDataset` class.

2. Run the training script:
   ```
   python train_cellular_automation.py
   ```

3. The script will train the model, save the best model, and generate visualizations.

4. Check the following output files:
   - `best_sand_model.pth`: The best performing model weights
   - `loss_plot.png`: A plot of training and validation losses
   - `prediction_comparison.png`: A visual comparison of input, true next frame, and predicted next frame

### Running the Trained Model

After training the model, you can interact with it using the `play.py` script:

1. Ensure you have a trained model (`best_sand_model.pth`) in your project directory.

2. Run the play script:
   ```
   python play.py
   ```

3. Follow the on-screen instructions to interact with the model and see predictions for falling sand simulations.

## Customization

You can modify the following parameters in the `train_cellular_automation.py` file:
- `num_epochs`: Number of training epochs
- Learning rate and scheduler parameters in the `train_model` function
- Model architecture in the `CellularAutomatonCNN` class
- Batch size and other DataLoader parameters

## Model Architecture

The model uses a series of convolutional layers with batch normalization:
1. Input layer: 7 channels (one-hot encoded cell types)
2. Hidden layers: 64 -> 128 -> 64 channels
3. Output layer: 7 channels (predicted cell types)

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

Copyright (C) 2023 David Hamner

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Acknowledgments

This project adapts convolutional neural networks for predicting falling sand simulations.
Claude AI (https://www.anthropic.com) was used for coding assistance and README generation.
