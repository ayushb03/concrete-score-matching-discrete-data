# Concrete Score Matching (CSM) Implementation

## Overview
This repository contains an implementation of **Concrete Score Matching (CSM)** based on the paper:

> *Concrete Score Matching: Generalized Score Matching for Discrete Data*  
> **Authors:** Chenlin Meng, Kristy Choi, Jiaming Song, Stefano Ermon  
> **Paper:** [arXiv:2211.00802](https://arxiv.org/abs/2211.00802)  

CSM is a novel approach to score matching in discrete spaces, addressing the limitations of traditional score-based models which rely on continuous gradients. It introduces the **Concrete Score**, which defines local directional changes of probabilities in discrete domains based on a predefined neighborhood structure.

## Features
- **Synthetic Data Generation**: Supports 1D and 2D discrete data generation.
- **Concrete Score Model**: Implements an embedding-based model for learning Concrete scores.
- **Concrete Score Matching Loss**: Custom training objective for score matching.
- **Metropolis-Hastings Sampling**: Utilizes the learned score model for data generation.
- **Visualization**: Generates histograms and scatter plots to analyze distributions.

## Installation
### Dependencies
Ensure you have Python 3.8+ and install required dependencies:
```bash
uv pip install torch numpy matplotlib scikit-learn torchvision
```

## Usage
### Running the Implementation
Execute the script to train the model and generate synthetic data samples:
```bash
uv run main.py
```

### Configuration
Modify the `Config` class to adjust parameters such as:
- `num_categories`: Number of discrete states.
- `data_structure`: "cycle" or "grid" neighborhood structure.
- `learning_rate`, `num_epochs`: Training parameters.
- `num_samples`, `num_steps`: Sampling parameters.

### Output
- **Training Progress**: Loss values printed per epoch.
- **Generated Figures**:
  - `figures/1d_distributions.png`: True vs. generated 1D data distribution.
  - `figures/2d_distributions.png`: True vs. generated 2D data scatter plot.

## Code Structure
- `config.py`: Configuration settings.
- `data.py`: Functions for generating synthetic data.
- `model.py`: Implementation of the Concrete Score Model and U-Net.
- `train.py`: Training loop with Concrete Score Matching loss.
- `sampling.py`: Metropolis-Hastings algorithm for sampling.
- `main.py`: Orchestrates data generation, training, and sampling.

## Citation
If you find this implementation useful, consider citing the original paper:
```bibtex
@article{meng2022concrete,
  title={Concrete Score Matching: Generalized Score Matching for Discrete Data},
  author={Meng, Chenlin and Choi, Kristy and Song, Jiaming and Ermon, Stefano},
  journal={arXiv preprint arXiv:2211.00802},
  year={2022}
}
```

## License
MIT License. See `LICENSE` for details.

## Acknowledgments
This implementation is inspired by the original work and aims to provide an accessible version for researchers and developers.

