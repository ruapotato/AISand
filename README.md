# Falling Sand Diffusion Model

This project implements a diffusion model for predicting the next frame in a falling sand simulation game. It uses PyTorch to create and train a neural network that can learn the physics of particle interactions in a 2D grid.

## Features

- Custom dataset loader for falling sand simulation frames
- Simplified diffusion model architecture
- Training loop with diffusion loss
- Inference function for predicting the next frame
- Visualization of results

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- Pillow

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/falling-sand-diffusion.git
   cd falling-sand-diffusion
   ```

2. Install the required packages:
   ```
   pip install torch torchvision matplotlib pillow
   ```

## Usage

1. Prepare your dataset:
   ```
   python gather.py
   ```

2. Run the training script:
   ```
   python train_diffusion.py
   ```

3. The script will train the model, save it, and generate a sample prediction.

4. Check the `frame_prediction_comparison.png` file to see a comparison between an input frame and the predicted next frame.

## Customization

You can modify the following parameters in the `train_diffusion.py` file:

- `epochs`: Number of training epochs
- `lr`: Learning rate for the optimizer
- `batch_size`: Batch size for training
- Model architecture in the `SimpleDiffusionModel` class

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

This project is inspired by the GameNGen paper and adapts the concept of diffusion models for falling sand simulations.
https://claude.ai/ was used also used for coding assistance. 
