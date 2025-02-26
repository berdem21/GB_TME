

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Example usage of SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate a random regression problem
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVR model
svr_model = SVR()

# Train the model
svr_model.fit(X_train, y_train)

# Make predictions
y_pred = svr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

import keras 

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
##############################
#applying basic data augmentation augmentation techniques to image data set
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt

# Create an instance of the ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load a sample image
img_path = 'path_to_your_image.jpg'  # Replace with the path to your cell image
img = load_img(img_path)  # Load image
x = img_to_array(img)  # Convert image to numpy array
x = x.reshape((1,) + x.shape)  # Reshape image

# Generate augmented images and display them
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0].astype('uint8'))
    i += 1
    if i % 4 == 0:  # Display 4 augmented images
        break

plt.show()

##############################
#adding noise to image data for data augmentation 

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt

# Custom function to add random noise to images
def add_noise(img):
    noise_factor = 0.2
    noise = np.random.randn(*img.shape) * noise_factor
    img_noisy = img + noise
    img_noisy = np.clip(img_noisy, 0., 1.)
    return img_noisy

# Create an instance of the ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=add_noise  # Add custom noise function
)

# Load a sample image
img_path = 'path_to_your_image.jpg'  # Replace with the path to your cell image
img = load_img(img_path)  # Load image
x = img_to_array(img) / 255.0  # Convert image to numpy array and normalize
x = x.reshape((1,) + x.shape)  # Reshape image

# Generate augmented images and display them
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 4 == 0:  # Display 4 augmented images
        break

plt.show()