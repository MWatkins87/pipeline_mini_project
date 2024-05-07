import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def r2_adj(x, y, model):
    r2 = model.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def filter_the_data(data_file):
    data = pd.read_csv(data_file)

    data.dropna(inplace=True)
    select_features = ["targeted_productivity", "smv", "wip"]
    filtered = data[select_features]	
    scaled = StandardScaler().fit_transform(filtered)
    scaled_df = pd.DataFrame(scaled, columns=select_features)
    y = data["actual_productivity"]
    X_full_train, X_full_test, y_train, y_test = train_test_split(scaled_df,  y)
    # Create the models
    lr1 = LinearRegression()
    # Fit the first model to the full training data. 
    lr1.fit(X_full_train, y_train)
    predicted1 = lr1.predict(X_full_test)
    # Score the predictions with mse and r2
    mse1 = mean_squared_error(y_test, predicted1)
    r21 = r2_score(y_test, predicted1)
    print(f"All Features:")
    print(f"mean squared error (MSE): {mse1}")
    print(f"R-squared (R2): {r21}")
    adj_score1 = r2_adj(X_full_test, y_test, lr1)
    print(f"Adjusted R2: {adj_score1}")
    cv_scores = cross_val_score(LinearRegression(), X_full_train, y_train, scoring = "r2")
    return cv_scores
    