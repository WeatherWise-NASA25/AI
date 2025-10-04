import pandas as pd
import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# --- 1. Load and Merge Data ---
def load_and_merge_data(data_path='.'):
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    df_list = [pd.read_csv(f) for f in csv_files if 'prediction' not in os.path.basename(f).lower()]
    if not df_list:
        raise ValueError("No CSV files found.")
    return pd.concat(df_list, ignore_index=True).sort_values(by=['city', 'date']).reset_index(drop=True)

# --- 2. Preprocessing & Feature Engineering ---
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['season'] = df['month'].apply(get_season)

    # --- Add Cyclical Features ---
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)

    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# --- 4. Model Building ---
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes))
    def forward(self, x): return self.network(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# --- 5. Data Preparation for Models ---
def create_sequences_by_city(features_df, labels_df, preprocessor, sequence_length=7):
    """Creates sequences on a per-city basis to avoid data leakage between cities."""
    all_xs, all_ys = [], []

    # Process each city independently
    for city in features_df['city'].unique():
        city_features_df = features_df[features_df['city'] == city]
        city_labels_df = labels_df.loc[city_features_df.index]

        # Preprocess the data for this city
        city_features_processed = preprocessor.transform(city_features_df)
        city_labels = city_labels_df.values

        # Create sequences for this city
        if len(city_features_processed) > sequence_length:
            xs, ys = [], []
            for i in range(len(city_features_processed) - sequence_length):
                xs.append(city_features_processed[i:(i + sequence_length)])
                ys.append(city_labels[i + sequence_length])
            all_xs.extend(xs)
            all_ys.extend(ys)
    
    return np.array(all_xs), np.array(all_ys)

# --- 6. Training & Evaluation ---
def train_model(model, loader, criterion, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate_model(model, loader, target_names, model_name):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            all_preds.append(model(inputs).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)
    print(f"\n--- {model_name} Evaluation ---")
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
        print(f'\nResults for: {name}')
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

# --- 9. Prediction Function ---
def predict_weather(city, date_str, full_df, export_to_csv=False):
    print(f"\n--- Making Prediction for {city} on {date_str} ---")
    preprocessor = joblib.load('preprocessor.joblib')
    target_columns = ['temperature_c', 'humidity_percent', 'precipitation_mm']

    # --- Create a realistic input based on historical averages ---
    pred_date = pd.to_datetime(date_str)
    month = pred_date.month
    
    # Filter historical data for the same city and month
    historical_data = full_df[(full_df['city'] == city) & (full_df['month'] == month)]
    if historical_data.empty:
        print(f"Warning: No historical data for {city} in month {month}. Using global average.")
        historical_data = full_df[full_df['month'] == month]
        if historical_data.empty:
             historical_data = full_df # Fallback to all data

    # Create an input row with mean values
    # Select only numeric columns for averaging
    numeric_cols = historical_data[preprocessor.feature_names_in_].select_dtypes(include=np.number).columns
    input_df = historical_data[numeric_cols].mean().to_frame().T
    input_df['city'] = city
    input_df['season'] = get_season(month)
    input_df['year'] = pred_date.year
    input_df['month'] = month
    input_df['day'] = pred_date.day

    # Ensure all columns are in the correct order
    input_df = input_df[preprocessor.feature_names_in_]

    # --- Load Model and Predict ---
    input_size = preprocessor.transform(input_df).shape[1]
    mlp_model = MLP(input_size, len(target_columns))
    mlp_model.load_state_dict(torch.load('mlp_weather_model.pth'))
    mlp_model.eval()

    processed_input = preprocessor.transform(input_df)
    input_tensor = torch.tensor(processed_input, dtype=torch.float32)

    with torch.no_grad():
        prediction = mlp_model(input_tensor).numpy().flatten()

    results = {label: f"{value:.2f}" for label, value in zip(target_columns, prediction)}
    print("MLP Predictions (based on historical averages):", results)

    if export_to_csv:
        pred_df = pd.DataFrame([results])
        pred_df['city'] = city
        pred_df['date'] = date_str
        pred_df.to_csv('predictions.csv', index=False)
        print("Predictions exported to predictions.csv")
    return results

if __name__ == "__main__":
    try:
        df = load_and_merge_data()
        df = preprocess_data(df)

        # Define features and targets for regression
        target_columns = ['temperature_c', 'humidity_percent', 'precipitation_mm']
        # Features are all columns except the targets and the original date string
        features_df = df.drop(columns=target_columns + ['date'])
        labels_df = df[target_columns]
        
        # Identify numeric and categorical features for preprocessing
        numeric_features = features_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = ['city', 'season']
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

        # Split data into training and testing sets
        X_train_df, X_test_df, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state=42)

        # Fit the preprocessor on the training data and transform both sets
        X_train_processed = preprocessor.fit_transform(X_train_df)
        X_test_processed = preprocessor.transform(X_test_df)
        joblib.dump(preprocessor, 'preprocessor.joblib')

        # --- MLP Model ---
        print("\n--- Training MLP Regression Model ---")
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_processed, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32)), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_test_processed, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)), batch_size=64)
        
        mlp_model = MLP(X_train_processed.shape[1], len(target_columns))
        train_model(mlp_model, train_loader, nn.MSELoss(), optim.Adam(mlp_model.parameters(), lr=0.001))
        evaluate_model(mlp_model, test_loader, target_columns, 'MLP')
        torch.save(mlp_model.state_dict(), 'mlp_weather_model.pth')
        print("\nMLP regression model saved.")

        # --- LSTM Model ---
        print("\n--- Training LSTM Regression Model ---")
        # Create sequences on a per-city basis
        X_seq, y_seq = create_sequences_by_city(features_df, labels_df, preprocessor)
        
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        train_loader_seq = DataLoader(TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32)), batch_size=64, shuffle=True)
        test_loader_seq = DataLoader(TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq, dtype=torch.float32)), batch_size=64)

        lstm_model = LSTMModel(input_size=X_train_seq.shape[2], hidden_size=64, num_layers=2, num_classes=len(target_columns))
        train_model(lstm_model, train_loader_seq, nn.MSELoss(), optim.Adam(lstm_model.parameters(), lr=0.001))
        evaluate_model(lstm_model, test_loader_seq, target_columns, 'LSTM')
        torch.save(lstm_model.state_dict(), 'lstm_weather_model.pth')
        print("\nLSTM regression model saved.")

        # --- 9. Example Prediction ---
        # Pass the original dataframe to the function to calculate historical averages
        predict_weather(city='New_York', date_str='2025-07-20', full_df=df, export_to_csv=True)

    except Exception as e:
        print(f"An error occurred: {e}")