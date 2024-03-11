import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the KPIs as features
features = ['Gestational age', 'Birthwt', 'FFn', 'CRP', 'PIGF', 'Cervicallngt',
            'BMI', 'MaternalEd', 'MHC', 'MN', 'SU', 'MG', 'SS', 'IM']

# Generate sample data
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, len(features))
y = np.random.randint(0, 2, n_samples)

# Create a DataFrame with the sample data
data = pd.DataFrame(X, columns=features)
data['Infant Mortality'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['Infant Mortality'], test_size=0.2, random_state=42)

# Train the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
