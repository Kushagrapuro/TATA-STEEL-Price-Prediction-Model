import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "C:\\Users\\Asus\\Downloads\\TATA Steel merged_dataset_finalized.xlsx"
df = pd.read_excel(file_path)

# Drop 'ticker' column
df = df.drop(columns=['ticker'])

# One-hot encode the 'district' column
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
district_encoded = one_hot_encoder.fit_transform(df[['district']])
df_encoded = pd.DataFrame(district_encoded, columns=one_hot_encoder.get_feature_names_out(['district']))
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(columns=['district'])

# Convert 'DATE' column to datetime
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y')

# Drop duplicate dates
df = df.drop_duplicates(subset=['DATE'])

# Sort by date and set the frequency to monthly
df = df.set_index('DATE')
df = df.sort_index()

# If there are any missing dates, forward-fill them
df = df.asfreq('M', method='ffill')  # Assuming monthly data

# ARIMA model training on 'close' column
model = ARIMA(df['close'], order=(5, 1, 0))  # Adjust (p,d,q) based on analysis
model_fit = model.fit()

# Prediction function with user input
def predict_steel_price():
    # Taking input from the user
    energy_req = float(input("Enter Energy Requirement: "))
    energy_avail = float(input("Enter Energy Availability: "))
    open_price = float(input("Enter Open Price: "))
    high_price = float(input("Enter High Price: "))
    low_price = float(input("Enter Low Price: "))
    volume = int(input("Enter Volume: "))
    
    # Creating a new sample with the user inputs
    new_sample = pd.DataFrame({
        'Energy Requirement': [energy_req],
        'Energy Availability': [energy_avail],
        'open': [open_price],
        'high': [high_price],
        'low': [low_price],
        'volume': [volume]
    })
    
    # One-hot encode the new sample for district (this depends on what district you want to predict for)
    # Assuming all district columns are 0 for simplicity (adjust based on your use case)
    district_cols = pd.DataFrame(np.zeros((1, len(one_hot_encoder.get_feature_names_out(['district'])))), 
                                 columns=one_hot_encoder.get_feature_names_out(['district']))
    
    new_sample = pd.concat([new_sample, district_cols], axis=1)
    
    # Making prediction
    predicted = model_fit.forecast(steps=1)[0]
    print(f"\nPredicted Close Price: {predicted:.2f}")
    
    # Calculate WMSE (Weighted Mean Squared Error)
    wmse = np.average((df['close'] - model_fit.fittedvalues) ** 2, weights=np.arange(len(df)))
    print(f"WMSE: {wmse:.2f}")
    
    # Compare predicted value with previous close price
    prev_close = df['close'].iloc[-1]
    if predicted > prev_close:
        print("Steel price is expected to be HIGH compared to the last recorded value.")
    else:
        print("Steel price is expected to be LOW compared to the last recorded value.")

# Call the prediction function
predict_steel_price()

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['close'], label='Actual')
plt.plot(df.index, model_fit.fittedvalues, color='red', label='Fitted')
plt.title('Actual vs Fitted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
