

import tensorflow as tf
import os
import glob
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2



# Function to parse VOC annotation XML files
def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects_count = {'car': 0, 'pedestrian': 0, 'bike': 0}
    for obj in root.findall('object'):
        label_elem = obj.find('name')
        if label_elem is not None:
            label = label_elem.text.strip().lower()
            if label in objects_count:
                objects_count[label] += 1
            else:
                print(f"Unknown label found: {label}")
        else:
            print("No 'name' tag found for object.")
    return objects_count

# Function to process images and count labels
def process_image(jpeg_file, image_paths, labels, class_counts, data_dir):
    image_name = os.path.basename(jpeg_file)
    annotation_file = os.path.join(data_dir, image_name[:-4] + '.xml')
    if os.path.exists(annotation_file):
        objects_count = parse_voc_annotation(annotation_file)
        image_paths.append(jpeg_file)
        labels.append(objects_count)
        for key, count in objects_count.items():
            class_counts[key] += count
    else:
        print(f"Annotation file not found for image: {image_name}")

# Function to load dataset
def load_dataset(data_dir):
    image_paths = []
    labels = []
    class_counts = {'car': 0, 'pedestrian': 0, 'bike': 0}
    sub_dirs = ['bike', 'car', 'pedestrian']
    for sub_dir in sub_dirs:
        subdir_path = os.path.join(data_dir, sub_dir)
        jpeg_files = glob.glob(os.path.join(subdir_path, '*.jpg'))
        for jpeg_file in jpeg_files:
            process_image(jpeg_file, image_paths, labels, class_counts, subdir_path)
    return image_paths, labels, class_counts

# Function to split dataset into training, validation, and test sets
def split_dataset(image_paths, labels, train_pct=0.6, val_pct=0.2):
    zipped_list = list(zip(image_paths, labels))
    np.random.shuffle(zipped_list)
    total_images = len(zipped_list)
    train_size = int(total_images * train_pct)
    val_size = int(total_images * val_pct)

    train_set = zipped_list[:train_size]
    val_set = zipped_list[train_size:train_size + val_size]
    test_set = zipped_list[train_size + val_size:]

    return train_set, val_set, test_set
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0  # Normalize the image to [0, 1]
    return img

def determine_label_index(label_dict):
    # Assuming label_dict is a dictionary like {'car': 0, 'pedestrian': 1, 'bike': 2}
    if label_dict['car'] > label_dict['pedestrian'] and label_dict['car'] > label_dict['bike']:
        return 0
    elif label_dict['pedestrian'] > label_dict['car'] and label_dict['pedestrian'] > label_dict['bike']:
        return 1
    else:
        return 2

def data_generator(dataset, batch_size=32):
    """ Generate batches of data from the dataset """
    while True:  # Loop forever so the generator never terminates
        np.random.shuffle(dataset)  # Shuffle the data for good measure.
        for i in range(0, len(dataset), batch_size):
            batch_items = dataset[i:i + batch_size]
            if not batch_items:
                continue  # In case the last batch has fewer items
            batch_images = [preprocess_image(img_path) for img_path, _ in batch_items]
            batch_labels = [determine_label_index(label_dict) for _, label_dict in batch_items]
            yield np.array(batch_images), np.array(batch_labels)

# Example usage in your existing setup:

# Function to plot class distribution
def plot_class_distribution(class_counts, title="Class Distribution"):
    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    plt.figure(figsize=(8, 4))
    plt.bar(classes, counts, color=['red', 'green', 'blue'])
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.show()

# Main block to execute the functions
if __name__ == "__main__":
    # Define the data directory
    data_dir = "/Users/sheshta/Desktop/group1dataset3"

    # Load the dataset and compute initial class counts
    image_paths, labels, class_counts = load_dataset(data_dir)

    # Plot the total number of images per class
    plot_class_distribution(class_counts, "Total Images per Class")

    # Split the dataset into training, validation, and testing sets
    train_set, val_set, test_set = split_dataset(image_paths, labels)
    train_gen = data_generator(train_set)
    val_gen = data_generator(val_set)
    test_gen = data_generator(test_set)
    # Optionally, you can further process or print out details about the dataset
    print("Total training images:", len(train_set))
    print("Total validation images:", len(val_set))
    print("Total testing images:", len(test_set))




initializers = {
     #'xavier_uniform': tf.keras.initializers.GlorotUniform(),
     #'xavier_normal': tf.keras.initializers.GlorotNormal(),
    'kaiming_uniform': tf.keras.initializers.HeUniform(),
    #'kaiming_normal': tf.keras.initializers.HeNormal()
}

'''
class CNNModel(tf.keras.Model):
    def __init__(self, initializer):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, padding='same')
        self.pool1 = MaxPooling2D(2, 2)
        self.conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, padding='same')
        self.pool2 = MaxPooling2D(2, 2)
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu', kernel_initializer=initializer)
        self.dropout = Dropout(0.3)
        self.out = Dense(3, activation='softmax', kernel_initializer=initializer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.out(x)
'''
class CNNModel(tf.keras.Model):
    def __init__(self, initializer, l2_strength=0.01):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, padding='same', kernel_regularizer=l2(l2_strength))
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling2D(2, 2)
        self.conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, padding='same', kernel_regularizer=l2(l2_strength))
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling2D(2, 2)
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2(l2_strength))
        self.bn3 = BatchNormalization()
        self.dropout = Dropout(0.3)
        self.out = Dense(3, activation='softmax', kernel_initializer=initializer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x, training=training)
        x = self.dropout(x, training=training)
        return self.out(x)


def train_model(model, train_gen, val_gen, optimizer, epochs, steps_per_epoch, validation_steps, lr_schedule, loss_fn):
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        val_loss = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop
        for step in range(steps_per_epoch):
            try:
                batch_images, batch_labels = next(train_gen)
            except StopIteration:
                # Reset the generator if it runs out of data
                train_gen = data_generator(train_set, batch_size)
                batch_images, batch_labels = next(train_gen)

            with tf.GradientTape() as tape:
                predictions = model(batch_images, training=True)
                loss = loss_fn(batch_labels, predictions)

            # Calculate the learning rate for the current step and set it
            lr = lr_schedule(step + epoch * steps_per_epoch)  # Compute the learning rate
            tf.keras.backend.set_value(optimizer.lr, lr)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss.update_state(loss)
            train_accuracy.update_state(batch_labels, predictions)

        # Validation loop
        for step in range(validation_steps):
            try:
                batch_images, batch_labels = next(val_gen)
            except StopIteration:
                # Reset the generator if it runs out of data
                val_gen = data_generator(val_set, batch_size)
                batch_images, batch_labels = next(val_gen)

            predictions = model(batch_images, training=False)
            v_loss = loss_fn(batch_labels, predictions)

            val_loss.update_state(v_loss)
            val_accuracy.update_state(batch_labels, predictions)

        # Store results for the epoch
        history['loss'].append(train_loss.result().numpy())
        history['accuracy'].append(train_accuracy.result().numpy())
        history['val_loss'].append(val_loss.result().numpy())
        history['val_accuracy'].append(val_accuracy.result().numpy())

        # Print metrics at the end of the epoch
        print(f"Train Loss: {train_loss.result().numpy()}, Train Accuracy: {train_accuracy.result().numpy()}")
        print(f"Val Loss: {val_loss.result().numpy()}, Val Accuracy: {val_accuracy.result().numpy()}")

    return history


def test_model(model, test_gen, test_steps, class_names=['car', 'pedestrian', 'bike']):
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for _ in range(test_steps):
        batch_images, batch_labels = next(test_gen)
        predictions = model(batch_images, training=False)  # ensure model is in inference mode
        test_accuracy.update_state(batch_labels, predictions)

        # Convert predictions to class indices
        predicted_class_indices = np.argmax(predictions, axis=1)
        actual_class_indices = batch_labels

        for actual, predicted in zip(actual_class_indices, predicted_class_indices):
            if actual == predicted:
                class_correct[class_names[actual]] += 1
            class_total[class_names[actual]] += 1

    overall_accuracy = test_accuracy.result()
    class_accuracies = {class_name: (class_correct[class_name] / class_total[class_name])
                        for class_name in class_names if class_total[class_name] > 0}

    return overall_accuracy, class_accuracies



# Define batch sizes and epochs
batch_sizes = [8]
epochs = 1
results = {}

# Define learning rate schedules
lr_schedules = {
    #'multistep': tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #boundaries=[3 * (len(train_set) // 32), 8 * (len(train_set) // 32), 12 * (len(train_set) // 32)],
        #values=[0.001, 0.0005, 0.0001, 0.00001]),
     'exponential': tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=10 * (len(train_set) // 32),
        decay_rate=0.9)
}

def accuracy_per_class(model, gen, steps, class_labels):
    correct_count = {label: 0 for label in class_labels}
    total_count = {label: 0 for label in class_labels}

    for _ in range(steps):
        images, labels = next(gen)
        predictions = model(images, training=False)
        predicted_classes = np.argmax(predictions.numpy(), axis=1)
        actual_classes = [np.argmax(label) for label in labels]

        for actual, predicted in zip(actual_classes, predicted_classes):
            actual_label = class_labels[actual]
            if actual == predicted:
                correct_count[actual_label] += 1
            total_count[actual_label] += 1

    class_accuracies = {label: correct_count[label] / total_count[label] for label in class_labels if total_count[label] > 0}
    return class_accuracies

# Define class labels
class_labels = ['car', 'pedestrian', 'bike']

for init_name, init_func in initializers.items():
    for batch_size in batch_sizes:
        steps_per_epoch = len(train_set) // batch_size
        validation_steps = len(val_set) // batch_size

        if len(train_set) % batch_size != 0:
            steps_per_epoch += 1  # Account for the last batch that might be smaller than the batch size

            train_steps_per_epoch = len(train_set) // batch_size
            val_steps_per_epoch = len(val_set) // batch_size


        for lr_name, lr_schedule in lr_schedules.items():
            print(f"Training with {init_name}, batch size {batch_size}, lr_schedule {lr_name}, epochs {epochs}")
            l2_strength = 0.01
            model = CNNModel(initializer=init_func, l2_strength=l2_strength)
            optimizer = Adam(learning_rate=lr_schedule)

            # Train the model
            history = train_model(model, train_gen, val_gen, optimizer, epochs, steps_per_epoch, validation_steps, lr_schedule, tf.keras.losses.SparseCategoricalCrossentropy())

            # Store the training results
            results[(init_name, batch_size, lr_name)] = history

            # Evaluate the model on the test set
            test_accuracy, per_class_accuracy = test_model(model, test_gen, len(test_set) // batch_sizes[0])
            print(f"Overall Test Accuracy: {test_accuracy.numpy()}")
            for class_name, accuracy in per_class_accuracy.items():
              print(f"Accuracy for {class_name}: {accuracy}")




# Define a dictionary for color mapping
# Existing color mapping dictionary
color_mapping = {
    ('Batch 8', 'LR multistep'): 'red',
    ('Batch 8', 'LR exponential'): 'blue',
    ('Batch 16', 'LR multistep'): 'green',
    ('Batch 16', 'LR exponential'): 'orange',
    ('Batch 32', 'LR multistep'): 'purple',
    ('Batch 32', 'LR exponential'): 'black'
}

# Example modification to ensure keys match during plotting


def plot_results(results, initializers):
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))  # Wider figure
    axes[0].set_title('Loss vs. Epochs')
    axes[1].set_title('Accuracy vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')

    for (init_name, batch_size, lr_name), history in results.items():
        key = (f'Batch {batch_size}', f'LR {lr_name}')
        color = color_mapping.get(key, 'gray')  # Default color if not found

        label = f'Batch {batch_size}, {lr_name}'

        # Plotting the train and validation curves
        axes[0].plot(history['loss'], label=f'{label} (Train)', color=color, linestyle='-')
        axes[0].plot(history['val_loss'], label=f'{label} (Val)', color=color, linestyle='--')
        axes[1].plot(history['accuracy'], label=f'{label} (Train)', color=color, linestyle='-')
        axes[1].plot(history['val_accuracy'], label=f'{label} (Val)', color=color, linestyle='--')

    # Place the legend above the plot
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4, fontsize='small')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4, fontsize='small')

    plt.subplots_adjust(top=0.85)  # Adjust the top margin to give more space for the legend
    plt.show()



plot_results(results, initializers)

from saliency.fullGrad import FullGrad
# Initialize FullGrad object
fullgrad = FullGrad(model)

# Check completeness property
# done automatically while initializing object
fullgrad.checkCompleteness()

image_path = "/Users/sheshta/Desktop/group1dataset3"
target_class = 0
image_tensor = preprocess_image(image_path)  # Ensure this returns image with batch size
# Obtain fullgradient decomposition
input_gradient, bias_gradients = fullgrad.fullGradientDecompose(image_tensor, target_class)

# Obtain saliency maps
saliency_map = fullgrad.saliency(image_tensor, target_class)
















