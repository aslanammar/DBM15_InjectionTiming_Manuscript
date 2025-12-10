import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from models import build_ann_model, build_cnn_model, build_lstm_model

def load_data(filepath, input_features, target_column):
    data = pd.read_excel(filepath)
    X = data[input_features].values
    y = data[[target_column]].values
    return X, y, data

def preprocess_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            X_scaler, y_scaler)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

def train_model(model, X_train, y_train, epochs=200, batch_size=32, patience=20):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def cross_validate(X, y, build_model_fn, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = build_model_fn(X_train_fold.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
        )
        
        model.fit(
            X_train_fold, y_train_fold,
            epochs=50, batch_size=16, validation_split=0.2,
            callbacks=[early_stopping], verbose=0
        )
        
        y_pred_fold = model.predict(X_val_fold, verbose=0)
        metrics = calculate_metrics(y_val_fold, y_pred_fold)
        
        for key in scores.keys():
            scores[key].append(metrics[key])
        
        print(f'Fold {fold}: R2={metrics["R2"]:.4f}, RMSE={metrics["RMSE"]:.4f}')
    
    print(f'\nMean R2: {np.mean(scores["R2"]):.4f} (±{np.std(scores["R2"]):.4f})')
    return scores

def main(data_path, input_features, target_column):
    X, y, data = load_data(data_path, input_features, target_column)
    
    (X_train, X_test, y_train, y_test, 
     X_scaler, y_scaler) = preprocess_data(X, y)
    
    models = {
        'ANN': build_ann_model,
        'CNN': build_cnn_model,
        'LSTM': build_lstm_model
    }
    
    results = {}
    
    for name, build_fn in models.items():
        print(f'\n{"="*50}')
        print(f'Training {name} Model')
        print("="*50)
        
        model = build_fn(X_train.shape[1])
        history = train_model(model, X_train, y_train)
        
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_test_orig = y_scaler.inverse_transform(y_test)
        
        metrics = calculate_metrics(y_test_orig, y_pred)
        results[name] = {'model': model, 'metrics': metrics, 'history': history}
        
        print(f'\n{name} Results:')
        for metric, value in metrics.items():
            print(f'  {metric}: {value:.4f}')
    
    print(f'\n{"="*50}')
    print('Model Comparison')
    print("="*50)
    
    for name, res in results.items():
        print(f'{name}: R2={res["metrics"]["R2"]:.4f}, RMSE={res["metrics"]["RMSE"]:.4f}')
    
    return results

if __name__ == "__main__":
    DATA_PATH = "data.xlsx"
    INPUT_FEATURES = ['CA', 'Angle']
    TARGET_COLUMN = 'target'
    
    results = main(DATA_PATH, INPUT_FEATURES, TARGET_COLUMN)
