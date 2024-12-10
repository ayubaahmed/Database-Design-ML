
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load the iris dataset
iris = load_iris()

# Define the size of the test set
test_size = 45

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=test_size, random_state=42)

# Define the values of n_neighbors to evaluate
n_values = [1, 5, 10]

# Evaluate the accuracy of the classifier for different values of n_neighbors
for n in n_values:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy for n_neighbors = {}: {:.7f}".format(n, accuracy))
