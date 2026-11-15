import pandas as pd  # handle tabular data
import matplotlib.pyplot as plt  # draw charts

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# Create the dataset
data = {
    "biodegradable": [1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
                      1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                      1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    "recyclable": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
                   0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
                   0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    "contaminated": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                     1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    "weight_grams": [120, 80, 150, 100, 25, 300, 40, 20, 15, 90,
                     30, 250, 18, 35, 110, 12, 75, 45, 22, 130,
                     95, 28, 50, 105, 16, 32, 85, 24, 55, 19],
    "waste_category": [
        "Organic", "Organic", "Organic", "Organic", "Recyclable",
        "Recyclable", "General", "Recyclable", "Recyclable", "Organic",
        "General", "Recyclable", "Recyclable", "Recyclable", "Organic",
        "Recyclable", "Organic", "General", "Recyclable", "Organic",
        "Organic", "Recyclable", "General", "Organic", "Recyclable",
        "Recyclable", "Organic", "Recyclable", "General", "Recyclable"
    ]
}

df = pd.DataFrame(data)
print("DATASET:")
print(df.head(10))
print("\\nDataset shape:", df.shape)


# Select features and target
features = ["biodegradable", "recyclable", "contaminated", "weight_grams"]

X = df[features]  # information used to make classification
y = df["waste_category"]  # category to be predicted


# Split data into training and testing portions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))


# Create and train the classification model
model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

print("\\nModel training completed!")


# Make predictions
predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual category": y_test.values,
    "Predicted category": predictions
})

print("\\nTEST RESULTS:")
print(results)


# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

print("\\nMODEL EVALUATION:")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))


# Plot the confusion matrix
labels = ["General", "Organic", "Recyclable"]
cm = confusion_matrix(y_test, predictions, labels=labels)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
display.plot(cmap="Blues")
plt.title("Waste Classification Model")
plt.tight_layout()
plt.show()


# For new predictions
def predict_waste_category(biodegradable, recyclable, contaminated, weight_grams):
    new_data = pd.DataFrame({
        "biodegradable": [biodegradable],
        "recyclable": [recyclable],
        "contaminated": [contaminated],
        "weight_grams": [weight_grams]
    })

    prediction = model.predict(new_data)
    return prediction[0]


print("\\n===== SRMIST WASTE CLASSIFICATION SYSTEM =====")

# Take inputs from the user
biodegradable = int(input("Is the waste biodegradable? (1 = Yes, 0 = No): "))
recyclable = int(input("Is the waste recyclable? (1 = Yes, 0 = No): "))
contaminated = int(input("Is the waste contaminated? (1 = Yes, 0 = No): "))
weight_grams = float(input("Enter approximate waste weight in grams: "))

# Generate prediction
result = predict_waste_category(
    biodegradable, recyclable, contaminated, weight_grams
)

print("\\n===== PREDICTION RESULT =====")
print("Predicted waste category:", result)
