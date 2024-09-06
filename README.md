# AISand

This project implements a Convolutional Neural Network (CNN) model for predicting the next frame in a falling sand simulation game. It uses PyTorch to create and train a neural network that can learn the physics of particle interactions in a 2D grid.

## Features

- Custom dataset loader for falling sand simulation frames with dynamic sequence selection
- CNN-based model architecture
- Training loop with cross-entropy loss and Adam optimizer
- Inference function for predicting the next frame
- Visualization of training progress and results
- Interactive Pygame script to run the trained model

## Requirements

- Python 3.7+
- PyTorch
- numpy
- Pillow
- tqdm
- psutil
- matplotlib
- Pygame

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ruapotato/AISand.git
   cd AISand
   ```
2. Install the required packages:
   ```
   pip install torch numpy pillow tqdm psutil matplotlib pygame
   ```

## Usage

### Training the Model

1. Prepare your dataset:
   ```
   ./gather.py
   ```
2. Run the training script:
   ```
   python train_v3.py
   ```
3. The script will train the model and save the best model as 'best_sand_model.pth'.

### Running the Trained Model

After training the model, you can interact with it using the `play_v3.py` script:

1. Ensure you have a trained model (`best_sand_model.pth`) in your project directory.
2. Run the play script:
   ```
   python play_v3.py
   ```
3. Use the mouse to draw particles and observe the model's predictions in real-time.
4. Use number keys 1-9 and 0, o, l, s to select different particle types.
5. Use the mouse wheel to adjust brush size.
6. Press 'c' to clear the screen.

## Customization

You can modify the following parameters in the training script:

- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `top_percent`: Percentage of most dynamic sequences to use (default: 0.25 for top 25%)
- Model architecture in the `SimpleSandModel` class
- Optimizer parameters in the `train_model` function

## Model Architecture

The model uses a series of convolutional layers:

1. Input layer: 14 channels (one-hot encoded cell types)
2. Hidden layers: 32 channels
3. Output layer: 14 channels (predicted cell types)

## Dataset Preprocessing

The dataset is preprocessed to focus on the most dynamic sequences:

1. Calculate the amount of movement for each sequence of frames.
2. Rank all sequences based on their movement.
3. Select the top 25% of sequences with the highest movement.
4. Use these selected sequences for training and validation.

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
