import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import logging

def balance_and_prepare_data(features, labels):
    """Apply SMOTE, reshape data, and perform train-test split."""
    logging.info(f'Initial features shape: {features.shape}')
    logging.info(f'Initial labels shape: {labels.shape}')
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(features, labels)
    
    logging.info(f'Resampled features shape: {X_resampled.shape}')
    logging.info(f'Resampled labels shape: {y_resampled.shape}')
    
    X_resampled = X_resampled.reshape((-1, 1, X_resampled.shape[1]))
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    logging.info(f"Training set class distribution: {np.bincount(y_train)}")
    logging.info(f"Validation set class distribution: {np.bincount(y_val)}")
    
    return X_train, X_val, y_train, y_val