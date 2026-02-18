import numpy as np
import kerastuner as kt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Activation, Lambda, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from utils import squeeze_last_dim, mean_over_time, expand_last_dim, weighted_multiplication
import logging

def attention_3d_block(inputs):
    """Attention mechanism block."""
    a = Dense(1, activation='tanh')(inputs)
    a = Lambda(squeeze_last_dim, output_shape=lambda s: (s[0], s[1]))(a)
    a = Activation('softmax')(a)
    a = Lambda(expand_last_dim)(a)
    weighted_inputs = Lambda(weighted_multiplication)([inputs, a])
    return weighted_inputs

def build_lstm_attention_model(hp):
    """Build LSTM model with attention for Keras Tuner."""
    inputs = Input(shape=(1, 1280))
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.6, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='log')
    
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(1e-4)))(inputs)
    attention_out = attention_3d_block(lstm_out)
    dropout_out = Dropout(dropout_rate)(attention_out)
    flatten_out = Lambda(mean_over_time)(dropout_out)
    dense_out = Dense(128, activation='relu')(flatten_out)
    dense_out = BatchNormalization()(dense_out)
    outputs = Dense(1, activation='sigmoid')(dense_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val, model_path):
    """Train the model with hyperparameter tuning."""
    tuner = kt.Hyperband(
        build_lstm_attention_model,
        objective='val_accuracy',
        max_epochs=100,
        factor=3,
        directory='my_dir',
        project_name='best_lstm_attention_model_tuning'
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=16, callbacks=[reduce_lr, early_stopping])
    
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = build_lstm_attention_model(best_hyperparameters)
    
    history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[reduce_lr, early_stopping])
    
    best_model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    y_pred_val = best_model.predict(X_val)
    y_pred_val_binary = (y_pred_val > 0.5).astype(int)
    logging.info("Classification Report:\n" + classification_report(y_val, y_pred_val_binary, target_names=['FAKE', 'REAL']))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    logging.info("Training curves saved as training_curves.png")