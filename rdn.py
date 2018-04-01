import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Data Loading Code Runs At This Point
    
# Load data
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and predictors
y = filtered_melbourne_data.Price
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_predictors]

#accuracy testing part
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

mel_model = RandomForestRegressor()
mel_model.fit(train_X,train_y)
preds_model = mel_model.predict(val_X)
print(mean_absolute_error(val_y,preds_model))