import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from torchvision import datasets, transforms
import os
from dataclasses import dataclass

# -------------------------
# Configuration
# -------------------------
@dataclass
class Config:
    # Data parameters
    num_categories: int = 16
    num_samples: int = 1000
    data_structure: str = "cycle"  # "cycle" or "grid"
    
    # Model parameters
    embedding_dim: int = 16
    hidden_dim: int = 64
    
    # Training parameters
    learning_rate: float = 0.001
    num_epochs: int = 1000
    print_interval: int = 1
    
    # Sampling parameters
    num_steps: int = 1000
    
    # Paths
    figures_dir: str = "figures"
    
    def __post_init__(self):
        # Create figures directory if it doesn't exist
        os.makedirs(self.figures_dir, exist_ok=True)

# -------------------------
# Helper Functions
# -------------------------
def generate_synthetic_data_1d(config):
    """
    Generate a synthetic 1D discrete dataset similar to the one in the paper.
    """
    data = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        size=config.num_samples,
        p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    )
    return torch.tensor(data, dtype=torch.long)

def generate_synthetic_data_2d(config):
    """
    Generate a synthetic 2D discrete dataset with multiple modes.
    """
    centers = [[0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25]]
    data, _ = make_blobs(n_samples=config.num_samples, centers=centers, cluster_std=0.05)
    data = (data * 90).astype(int)  # Quantize to 91x91 bins
    return torch.tensor(data, dtype=torch.long)

def get_neighbors(x, config):
    """
    Define the neighborhood structure for a 1D discrete space.
    """
    if config.data_structure == "cycle":
        neighbors = [(x - 1) % config.num_categories, (x + 1) % config.num_categories]
    elif config.data_structure == "grid":
        neighbors = []
        if x > 0:
            neighbors.append(x - 1)
        if x < config.num_categories - 1:
            neighbors.append(x + 1)
    return neighbors

def compute_concrete_score(p_data, x, neighbors):
    """
    Compute the Concrete score for a given input x.
    """
    scores = []
    for neighbor in neighbors:
        score = (p_data[neighbor] - p_data[x]) / p_data[x]
        scores.append(score)
    return torch.tensor(scores)

# -------------------------
# Model Definition
# -------------------------
class ConcreteScoreModel(nn.Module):
    def __init__(self, num_categories, embedding_dim=16):
        super(ConcreteScoreModel, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # Output dimension corresponds to neighbors
        )

    def forward(self, x):
        x_embedded = self.embedding(x)
        scores = self.fc(x_embedded)
        return scores

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# -------------------------
# Training Objective
# -------------------------
def concrete_score_matching_loss(model, data, config):
    """
    Implement the Concrete Score Matching (CSM) objective.
    """
    loss = 0.0
    for x in data:
        neighbors = get_neighbors(x.item(), config)
        true_scores = compute_concrete_score(p_data, x.item(), neighbors)
        predicted_scores = model(x)[:len(neighbors)]  # Predict scores for neighbors
        loss += ((predicted_scores - true_scores) ** 2).mean()
    return loss / len(data)

# -------------------------
# Sampling via Metropolis-Hastings
# -------------------------
def metropolis_hastings_sampling(model, config):
    """
    Perform sampling using the Metropolis-Hastings algorithm.
    """
    samples = []
    current_sample = torch.randint(0, config.num_categories, (1,)).item()
    for _ in range(config.num_samples):
        for _ in range(config.num_steps):
            neighbors = get_neighbors(current_sample, config)
            proposed_sample = np.random.choice(neighbors)
            # Compute acceptance probability
            current_score = model(torch.tensor([current_sample])).exp().sum()
            proposed_score = model(torch.tensor([proposed_sample])).exp().sum()
            acceptance_prob = min(1, (proposed_score / current_score).item())
            if np.random.rand() < acceptance_prob:
                current_sample = proposed_sample
        samples.append(current_sample)
    return torch.tensor(samples)

# -------------------------
# Main Script
# -------------------------
if __name__ == "__main__":
    # Initialize config
    config = Config()
    
    # Generate synthetic data
    data_1d = generate_synthetic_data_1d(config)
    data_2d = generate_synthetic_data_2d(config)

    # Compute empirical probabilities
    p_data = torch.zeros(config.num_categories)
    for x in data_1d:
        p_data[x] += 1
    p_data /= p_data.sum()

    # Initialize model and optimizer
    model = ConcreteScoreModel(num_categories=config.num_categories, 
                             embedding_dim=config.embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        loss = concrete_score_matching_loss(model, data_1d, config)
        loss.backward()
        optimizer.step()
        if epoch % config.print_interval == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Sampling using Metropolis-Hastings
    samples = metropolis_hastings_sampling(model, config)

    # Save plots
    # 1D Data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_1d.numpy(), bins=range(config.num_categories + 1), 
             align='left', rwidth=0.8, label="True Data")
    plt.title("True 1D Data Distribution")
    plt.xlabel("Category")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(samples.numpy(), bins=range(config.num_categories + 1), 
             align='left', rwidth=0.8, label="Generated Samples")
    plt.title("Generated 1D Samples")
    plt.xlabel("Category")
    plt.ylabel("Frequency")
    
    plt.savefig(os.path.join(config.figures_dir, '1d_distributions.png'))
    plt.close()

    # 2D Data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5, label="True Data")
    plt.title("True 2D Data Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.subplot(1, 2, 2)
    plt.scatter(samples.numpy(), np.zeros_like(samples.numpy()), alpha=0.5, label="Generated Samples")
    plt.title("Generated 1D Samples")
    plt.xlabel("Category")
    plt.ylabel("Frequency")
    
    plt.savefig(os.path.join(config.figures_dir, '2d_distributions.png'))
    plt.close()