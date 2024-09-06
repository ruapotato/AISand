# AISand

This project implements a Convolutional Neural Network (CNN) model for predicting the next frame in a falling sand simulation game. It uses PyTorch to create and train a neural network that can learn the physics of particle interactions in a 2D grid.

## Features

- Custom dataset loader for falling sand simulation frames
- CNN-based model architecture
- Training loop with cross-entropy loss and learning rate scheduling
- Inference function for predicting the next frame
- Visualization of training progress and results
- Interactive Pygame script to run the trained model

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- Pillow
- tqdm
- Pygame

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ruapotato/AISand.git
   cd AISand
   ```
2. Install the required packages:
   ```
   pip install torch torchvision numpy matplotlib pillow tqdm pygame
   ```

## Usage

### Training the Model

1. Prepare your dataset:
   ```
   ./gather.py
   ```
2. Run the training script:
   ```
   python train.py
   ```
3. The script will train the model and save the best model.

### Running the Trained Model

After training the model, you can interact with it using the `play_v2.py` script:

1. Ensure you have a trained model (`best_improved_sand_model.pth`) in your project directory.
2. Run the play script:
   ```
   python play.py
   ```
3. Use the mouse to draw particles and observe the model's predictions in real-time.
4. Use number keys 1-9 and 0, o, l, s to select different particle types.
5. Use the mouse wheel to adjust brush size.
6. Press 'c' to clear the screen.

## Customization

You can modify the following parameters in the training script:
- `num_epochs`: Number of training epochs
- Learning rate and scheduler parameters in the `train_model` function
- Model architecture in the `ImprovedSandModel` class
- Batch size and other DataLoader parameters

## Model Architecture

The improved model uses a series of convolutional layers with batch normalization:
1. Input layer: 14 channels (one-hot encoded cell types)
2. Hidden layers: 64 -> 128 -> 64 channels
3. Output layer: 14 channels (predicted cell types)

## Particle Types

The simulation includes the following particle types:
EMPTY, SAND, WATER, PLANT, WOOD, ACID, FIRE, STEAM, SALT, TNT, WAX, OIL, LAVA, STONE

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
Thanks to https://code.google.com/archive/p/fallingsand-python/
Claude AI (https://www.anthropic.com) was used for coding assistance and README generation.
