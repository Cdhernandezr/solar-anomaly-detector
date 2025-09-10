class Config:
    """Configuración centralizada del proyecto"""
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 15
    PATIENCE = 3
    RANDOM_SEED = 42

    # Proporciones de división del dataset
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.20
    TEST_RATIO = 0.10