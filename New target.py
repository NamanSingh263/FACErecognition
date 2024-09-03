import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Function to load dataset
def load_dataset(dataset_path):
    X = []
    y = []
    person_names = []
    person_id = 0

    # Iterate through each person's directory
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_dir):
            continue  # Skip if it's not a directory

        person_names.append(person_name)

        # Iterate through each image in the person's directory
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            if not image_path.endswith(('.jpg', '.jpeg', '.png')):
                continue  # Skip non-image files

            # Read and preprocess the image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip if image can't be read

            img = cv2.resize(img, (300, 300))  # Resize if necessary
            X.append(img.flatten())
            y.append(person_id)

        person_id += 1

    return np.array(X), np.array(y), np.array(person_names)


# Function to plot eigenfaces
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# Load dataset
dataset_path = "dataset/dataset/faces"
X, y, person_names = load_dataset(dataset_path)
n_samples, n_features = X.shape
n_classes = len(np.unique(y))

print(f"Number of samples: {n_samples}, Number of features: {n_features}, Number of classes: {n_classes}")

# Split data into training and test sets (60% training, 40% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 2: Perform PCA for dimensionality reduction
n_components = 150  # Adjust based on your dataset and computational limits
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Original data shape: {X_train.shape}")
print(f"PCA-transformed data shape: {X_train_pca.shape}")

# Step 6: Find the best directions (Generation of feature vectors)
eigenfaces = pca.components_.reshape((n_components, 300, 300))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, 300, 300)
plt.show()

# Step 9: Apply ANN for training

clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=0.001)

# Fit the model
clf.fit(X_train_pca, y_train)

# Predict on test data
y_pred = clf.predict(X_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plotting accuracy vs number of components (k)
k_values = range(1, n_components + 1)
accuracies = []
for k in k_values:
    pca = PCA(n_components=k, svd_solver='randomized', whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000, alpha=1e-4,
                        solver='adam', verbose=0, tol=1e-4, random_state=1,
                        learning_rate_init=0.001)

    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.plot(k_values, accuracies, 'bo-', linewidth=2)
plt.title('Accuracy vs Number of PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
