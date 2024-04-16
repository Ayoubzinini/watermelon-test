from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from pandas import read_excel ,DataFrame

# Prepare the data
"""
X = df.drop(columns=['Brix_Degree']).values  # Features (spectral data)
y = df['Brix_Degree'].values  # Target (Brix degree)
"""
#"""
db=read_excel("C:/Users/ayoub/OneDrive/Documents/GitHub/corn-project/data-corn-feuille-t0.xlsx").dropna()
X=X=db.drop(['Unnamed: 0','Y1','Y2'],axis=1)
y=db['Y1']
#"""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network architecture
def create_model(num_neurons, activation):
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=num_wavelengths, activation=activation))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create KerasRegressor for use in GridSearchCV
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define hyperparameters grid for tuning
param_grid = {
    'num_neurons': [32, 64, 128],  # Number of neurons in hidden layer
    'activation': ['relu', 'tanh', 'sigmoid']  # Activation function
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
print("Best Hyperparameters: ", grid_result.best_params_)

# Train the model with the best hyperparameters
best_model = grid_result.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = best_model.score(X_train_scaled, y_train)
test_score = best_model.score(X_test_scaled, y_test)

print("Train Score:", train_score)
print("Test Score:", test_score)
