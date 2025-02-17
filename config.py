class Config:
    # Data parameters
    NUM_CATEGORIES = 16
    NUM_SAMPLES = 10
    EMBEDDING_DIM = 16
    
    # Model parameters
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    ACTIVATION = 'tanh'
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 2
    PRINT_INTERVAL = 1
    
    # Sampling parameters
    NUM_STEPS = 10
    STRUCTURE = "cycle"
    
    # Data generation parameters
    DATA_1D_PROBS = [0.05]*6 + [0.1]*4 + [0.05]*6
    DATA_2D_CENTERS = [[0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25]]
    DATA_2D_CLUSTER_STD = 0.05
    DATA_2D_QUANTIZATION = 90 