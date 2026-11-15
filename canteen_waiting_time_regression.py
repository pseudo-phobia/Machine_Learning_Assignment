import pandas as pd  #handle tabular data
import matplotlib.pyplot as plt #draw charts

from sklearn.model_selection import train_test_split #divide data into training and testing portions
from sklearn.linear_model import LinearRegression #prediction algorithm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {"queue_size": [5, 10, 15, 8, 20, 12, 6, 18, 25, 9, 14, 22, 7, 16, 11, 19, 4, 13, 24, 17, 9, 21, 6, 15, 12, 23, 8, 18, 10, 20], 
        "active_counters": [2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2, 2, 3, 2, 2], 
        "time_of_day": [10, 13, 13, 16, 13, 14, 11, 13, 13, 16, 12, 13, 10, 14, 12, 13, 9, 12, 13, 14, 16, 13, 11, 14, 12, 13, 10, 13, 11, 14], 
        "average_order_time": [1.0, 1.2, 1.5, 0.8, 1.7, 0.9, 1.1, 1.6, 1.0, 0.7, 1.3, 1.8, 0.8, 1.4, 1.0, 1.5, 0.9, 1.2, 1.7, 1.3, 0.8, 1.6, 0.9,
                            1.4, 1.1, 1.8, 0.7, 1.5, 1.0, 1.6], 
        "waiting_time": [3, 8, 12, 3, 16, 5, 4, 14, 8, 3, 9, 18, 2, 11, 5, 15, 2, 7, 19, 12, 4, 17, 3, 10, 6, 20, 3, 13, 5, 16]}


df = pd.DataFrame(data)
print("DATASET:")
print(df.head(10))
print("\nDataset shape:", df.shape)


features = ["queue_size", "active_counters", "time_of_day", "average_order_time"]

X = df[features]  #information used to make prediction
y = df["waiting_time"] #value to be predicted


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))


model = LinearRegression()
model.fit(X_train, y_train) #find best intercept and coefficients from data
print(f"\nThe Learned Linear Regression equation has intercept {model.intercept_} and coefficients {model.coef_}")
print("\nModel training completed!")


predictions = model.predict(X_test)
results = pd.DataFrame({"Actual waiting time": y_test.values, "Predicted waiting time": predictions.round(2)})
print("\nTEST RESULTS:")
print(results)


mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION:")
print("Mean Absolute Error:", round(mae, 2), "minutes")
print("Mean Squared Error:", round(mse, 2))
print("R-squared:", round(r2, 3))

# Plotting the graph
plt.figure(figsize=(7, 5))
plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
plt.xlabel("Actual waiting time (minutes)")
plt.ylabel("Predicted waiting time (minutes)")
plt.title("Canteen Waiting-Time Prediction")
plt.show()

# For new predictions
def predict_waiting_time(queue_size, active_counters, time_of_day, average_order_time):
    new_data = pd.DataFrame({"queue_size": [queue_size], "active_counters": [active_counters],
                             "time_of_day": [time_of_day], "average_order_time": [average_order_time]})
    prediction = model.predict(new_data)
    return round(max(0, prediction[0]), 2)



print("\n===== CANTEEN WAITING TIME PREDICTOR =====")

#Take inputs from the user
queue_size = int(input("Enter the number of people in the queue: "))
active_counters = int(input("Enter the number of active counters: "))
time_of_day = float(input("Enter the current hour: "))
average_order_time = float(input("Enter average order preparation time (minutes): "))

#Generate prediction
result = predict_waiting_time(queue_size, active_counters, time_of_day, average_order_time)
print("\n===== PREDICTION RESULT =====")
print("Estimated waiting time:", result, "minutes")