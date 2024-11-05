import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt


# Function to parse VOC annotation XML file
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

def process_image(jpeg_file, image_paths, labels, class_counts, data_dir):
    image_name = os.path.basename(jpeg_file)
    annotation_file = os.path.join(data_dir, image_name[:-4] + '.xml')
    if os.path.exists(annotation_file):
        objects_count = parse_voc_annotation(annotation_file)
        image_paths.append(jpeg_file)
        labels.append(objects_count)
        for key in objects_count:
            class_counts[key] += objects_count[key]
    else:
        print(f"Annotation file not found for image: {image_name}")

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

def data_generator(dataset, batch_size=32):
    while True:
        np.random.shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            batch_items = dataset[i:i+batch_size]
            batch_images = [preprocess_image(img_path) for img_path, _ in batch_items]
            batch_labels = [determine_label_index(label_dict) for _, label_dict in batch_items]
            yield np.array(batch_images), np.array(batch_labels)

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def determine_label_index(label_dict):
    if label_dict['car'] >= label_dict['pedestrian'] and label_dict['car'] >= label_dict['bike']:
        return 0
    elif label_dict['pedestrian'] >= label_dict['car'] and label_dict['pedestrian'] >= label_dict['bike']:
        return 1
    else:
        return 2

# Main data processing
data_dir = "/content/drive/MyDrive/final1.v1i.voc/data"  # Update this to your dataset directory
image_paths, labels, class_counts = load_dataset(data_dir)
train_set, val_set, test_set = split_dataset(image_paths, labels)

train_gen = data_generator(train_set)
val_gen = data_generator(val_set)
test_gen = data_generator(test_set)

# Example to process and print dataset sizes
print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Testing set size: {len(test_set)}")



initializers = {
    # 'xavier_uniform': tf.keras.initializers.GlorotUniform(),
    # 'xavier_normal': tf.keras.initializers.GlorotNormal(),
    'kaiming_uniform': tf.keras.initializers.HeUniform(),
    'kaiming_normal': tf.keras.initializers.HeNormal()
}


class CNNModel(tf.keras.Model):
    def __init__(self, initializer):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, padding='same')
        self.pool1 = MaxPooling2D(2, 2)
        self.conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, padding='same')
        self.pool2 = MaxPooling2D(2, 2)
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu', kernel_initializer=initializer)
        self.out = Dense(3, activation='softmax', kernel_initializer=initializer)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.out(x)


def train_model(model, train_gen, val_gen, optimizer, epochs, steps_per_epoch, validation_steps, lr_schedule, loss_fn):
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    # Initialize the optimizer without a set learning rate
    optimizer = Adam()
    global_step = 0  # Initialize global step

    for epoch in range(epochs):
        # Initialize the metrics at the start of each epoch
        train_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        val_loss = tf.keras.metrics.Mean()
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop
        for step in range(steps_per_epoch):
            batch_images, batch_labels = next(train_gen)
            with tf.GradientTape() as tape:
                predictions = model(batch_images, training=True)
                loss = loss_fn(batch_labels, predictions)

            # Calculate the learning rate for the current step and set it
            lr = lr_schedule(step + epoch * steps_per_epoch)  # Compute the learning rate
            tf.keras.backend.set_value(optimizer.lr, lr)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update the metrics and global step
            train_loss.update_state(loss)
            train_accuracy.update_state(batch_labels, predictions)
            global_step += 1  # Increment global step

        # Validation loop
        for step in range(validation_steps):
            batch_images, batch_labels = next(val_gen)
            predictions = model(batch_images, training=False)
            v_loss = loss_fn(batch_labels, predictions)

            val_loss.update_state(v_loss)
            val_accuracy.update_state(batch_labels, predictions)

        # Append the metrics' results to the history
        history['loss'].append(train_loss.result().numpy())
        history['val_loss'].append(val_loss.result().numpy())
        history['accuracy'].append(train_accuracy.result().numpy())
        history['val_accuracy'].append(val_accuracy.result().numpy())

        # Print the metrics at the end of each epoch
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss.result().numpy()}, Train Accuracy: {train_accuracy.result().numpy()}")
        print(f"Val Loss: {val_loss.result().numpy()}, Val Accuracy: {val_accuracy.result().numpy()}")

    return history


def test_model(model, test_gen, test_steps):
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for _ in range(test_steps):
        batch_images, batch_labels = next(test_gen)
        predictions = model(batch_images, training=False)  # ensure model is in inference mode
        test_accuracy.update_state(batch_labels, predictions)

    return test_accuracy.result()

# Define batch sizes and epochs
batch_sizes = [8]
epochs = 2
results = {}

# Define learning rate schedules
lr_schedules = {
    'multistep': tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[3 * (len(train_set) // 32), 8 * (len(train_set) // 32), 12 * (len(train_set) // 32)],
        values=[0.001, 0.0005, 0.0001, 0.00001]),
    'exponential': tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=10 * (len(train_set) // 32),
        decay_rate=0.9)
}

# Train and evaluate models with different initializers and learning rate schedules
for init_name, init_func in initializers.items():
    for batch_size in batch_sizes:
        # Calculate steps per epoch based on batch size
        steps_per_epoch = len(train_set) // batch_size
        validation_steps = len(val_set) // batch_size

        for lr_name, lr_schedule in lr_schedules.items():
            print(f"Training with {init_name}, batch size {batch_size}, lr_schedule {lr_name}, epochs {epochs}")
            model = CNNModel(initializer=init_func)

            # Initialize the optimizer with the learning rate schedule
            optimizer = Adam(learning_rate=lr_schedule)

            # Train the model
            history = train_model(
                model, train_gen, val_gen, optimizer, epochs, steps_per_epoch, validation_steps,
                lr_schedule, tf.keras.losses.SparseCategoricalCrossentropy()
            )

            # Store the training results
            results[(init_name, batch_size, lr_name)] = history

            # Evaluate the model on the test set
            test_accuracy = test_model(model, test_gen, len(test_set) // batch_size)
            print(f"Test Accuracy with {init_name}, batch size {batch_size}, lr_schedule {lr_name}: {test_accuracy.numpy()}")





# Define a dictionary for color mapping
color_mapping = {
    ('Batch 8', 'LR multistep'): 'red',
    ('Batch 8', 'LR exponential'): 'blue',
    ('Batch 16', 'LR multistep'): 'green',
    ('Batch 16', 'LR exponential'): 'orange',
    ('Batch 32', 'LR multistep'): 'purple',
    ('Batch 32', 'LR exponential'): 'black'
}

# Function to plot the results
def plot_results(results, initializers):
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].set_title('Loss vs. Epochs')
    axes[1].set_title('Accuracy vs. Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    
    for (init_name, batch_size, lr_name), history in results.items():
        color = color_mapping[(f'Batch {batch_size}', lr_name)]
        label = f'{batch_size}, {lr_name}'
        
        # Loss plot
        axes[0].plot(history['loss'], label=f'{label} (Train)', color=color, linestyle='-')
        axes[0].plot(history['val_loss'], label=f'{label} (Val)', color=color, linestyle='--')
        
        # Accuracy plot
        axes[1].plot(history['accuracy'], label=f'{label} (Train)', color=color, linestyle='-')
        axes[1].plot(history['val_accuracy'], label=f'{label} (Val)', color=color, linestyle='--')

    # Shrink current axis's height by 10% on the bottom to make space for the legend outside the plot
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
    
    # Put a legend below current axis
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=4, fontsize='small')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=4, fontsize='small')
    
    plt.show()

plot_results(results, initializers)








