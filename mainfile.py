import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


RESIZE_SHAPE = (100, 250)

def save_resized_images(src_dir, dst_dir, shape=(100, 250)):
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(dst_dir, file), img_resized)


correct_dir = r'C:\Users\sandy\OneDrive\Desktop\PROJECT\speechocean762\correct'
incorrect_dir = r'C:\Users\sandy\OneDrive\Desktop\PROJECT\speechocean762\incorrect'

resized_correct_dir = r'C:\Users\sandy\OneDrive\Desktop\PROJECT\resized_correct'
resized_incorrect_dir = r'C:\Users\sandy\OneDrive\Desktop\PROJECT\resized_incorrect'


save_resized_images(correct_dir, resized_correct_dir, shape=RESIZE_SHAPE)
save_resized_images(incorrect_dir, resized_incorrect_dir, shape=RESIZE_SHAPE)

def load_resized_images(correct_dir, incorrect_dir):
    file_paths, labels = [], []


    for file in os.listdir(correct_dir):
        if file.endswith('.png'):
            file_paths.append(os.path.join(correct_dir, file))
            labels.append(1)  

   
    for file in os.listdir(incorrect_dir):
        if file.endswith('.png'):
            file_paths.append(os.path.join(incorrect_dir, file))
            labels.append(0)  

    return file_paths, labels

file_paths, labels = load_resized_images(resized_correct_dir, resized_incorrect_dir)


X_train_files, X_temp_files, y_train, y_temp = train_test_split(file_paths, labels, test_size=0.2, random_state=42)
X_val_files, X_test_files, y_val, y_test = train_test_split(X_temp_files, y_temp, test_size=0.5, random_state=42)

def process_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, RESIZE_SHAPE)
    image = image / 255.0  
    return image, label

def create_dataset(file_paths, labels, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 8
train_dataset = create_dataset(X_train_files, y_train, batch_size)
val_dataset = create_dataset(X_val_files, y_val, batch_size)
test_dataset = create_dataset(X_test_files, y_test, batch_size)

# CNN 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 250, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


steps_per_epoch = len(X_train_files) // 100

history = model.fit(
    train_dataset,
    epochs=12,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch
)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

def extract_features(dataset):
    features = []
    labels = []
    for image, label in dataset:
        feature = feature_extractor(image)
        features.append(feature.numpy())
        labels.append(label.numpy())
    return np.vstack(features), np.hstack(labels)

X_train_features, y_train_labels = extract_features(train_dataset)
X_test_features, y_test_labels = extract_features(test_dataset)

# SVM and KNN classifiers
svm_classifier = SVC(kernel='linear', probability=True)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

svm_classifier.fit(X_train_features, y_train_labels)
knn_classifier.fit(X_train_features, y_train_labels)

y_pred_svm = svm_classifier.predict(X_test_features)
y_pred_knn = knn_classifier.predict(X_test_features)

print("SVM Classification Report:")
print(classification_report(y_test_labels, y_pred_svm, target_names=['incorrect', 'correct']))

print("KNN Classification Report:")
print(classification_report(y_test_labels, y_pred_knn, target_names=['incorrect', 'correct']))

y_pred_cnn = model.predict(test_dataset)
y_pred_cnn_labels = (y_pred_cnn > 0.5).astype(int).flatten()  

print("CNN Classification Report:")
print(classification_report(y_test_labels, y_pred_cnn_labels, target_names=['incorrect', 'correct']))