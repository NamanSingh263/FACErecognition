import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import cv2


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


dir_name = "dataset/dataset/faces"
y = []
X = []
target_names = []
person_id = 0
h = w = 300
n_samples = 0
class_names = []

# Iterate through each person's directory
for person_name in os.listdir(dir_name):
    dir_path = os.path.join(dir_name, person_name)
    class_names.append(person_name)

    # Iterate through each image in the person's directory
    for image_name in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image_name)

        # Read and preprocess the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (h, w))
        v = resized_image.flatten()

        # Append the flattened image and labels
        X.append(v)
        y.append(person_id)
        n_samples += 1

    # Add person's name to target_names and increment person_id
    target_names.append(person_name)
    person_id += 1

# Convert lists to numpy arrays
y = np.array(y)
X = np.array(X)
target_names = np.array(target_names)
n_features = X.shape[1]

print("Number of samples:", n_samples)
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# PCA to extract eigenfaces
n_components = 150
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()

# Projecting data onto the eigenfaces basis
print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape, X_test_pca.shape)

# LDA for further dimensionality reduction
print("Computing Linear Discriminant Analysis (LDA)")
lda = LDA().fit(X_train_pca, y_train)
X_train_lda = lda.transform(X_train_pca)
X_test_lda = lda.transform(X_test_pca)
print("LDA projection shape:", X_train_lda.shape, X_test_lda.shape)

# Training MLP Classifier
clf = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 10), max_iter=1000, verbose=True).fit(X_train_lda, y_train)
print("Model Weights:")
model_info = [coef.shape for coef in clf.coefs_]
print(model_info)

# Predictions and accuracy calculation
y_pred = clf.predict(X_test_lda)
accuracy = np.mean(y_pred == y_test) * 100
print("Accuracy: %.2f%%" % accuracy)

# Plot results
prediction_titles = [f"Predicted: {target_names[pred]} \nActual: {target_names[actual]}" for pred, actual in
                     zip(y_pred, y_test)]
plot_gallery(X_test, prediction_titles, h, w)
plt.show()
