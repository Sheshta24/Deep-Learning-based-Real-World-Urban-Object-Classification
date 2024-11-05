import os
import glob
import xml.etree.ElementTree as ET

def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects_count = {
        'cars': 0,
        'pedestrians': 0,
        'bikes': 0
    }

    for obj in root.findall('object'):
        label_elem = obj.find('name')
        if label_elem is not None:
            label = label_elem.text.strip().lower()  # Normalize label (strip whitespace, convert to lowercase)
            if label in objects_count:
                objects_count[label] += 1
            else:
                print(f"Unknown label found: {label}")
        else:
            print("No 'name' tag found for object.")

    return objects_count

def load_dataset(data_dir):
    image_paths = []
    annotation_paths = []
    objects_counts = {
        'cars': 0,
        'pedestrians': 0,
        'bikes': 0
    }

    # List all JPEG files in the data_dir (train/valid/test)
    jpeg_files = glob.glob(os.path.join(data_dir, '*.jpg'))

    # Iterate over each JPEG file to find corresponding XML file
    for jpeg_file in jpeg_files:
        image_name = os.path.basename(jpeg_file)
        annotation_file = os.path.join(data_dir, image_name[:-4] + '.xml')

        # Check if corresponding XML annotation file exists
        if os.path.exists(annotation_file):
            image_paths.append(jpeg_file)
            annotation_paths.append(annotation_file)

            # Parse annotation file to count objects by class
            class_counts = parse_voc_annotation(annotation_file)
            for class_label, count in class_counts.items():
                objects_counts[class_label] += count
        else:
            print(f"Annotation file not found for image: {image_name}")

    return image_paths, annotation_paths, objects_counts

def load_separate_datasets(train_folder, valid_folder, test_folder):
    train_images, train_annotations, train_objects_counts = load_dataset(train_folder)
    val_images, val_annotations, val_objects_counts = load_dataset(valid_folder)
    test_images, test_annotations, test_objects_counts = load_dataset(test_folder)

    return (train_images, train_annotations, train_objects_counts), (val_images, val_annotations, val_objects_counts), (test_images, test_annotations, test_objects_counts)

# Path to training, validation, and test folders
train_folder = r"C:\Users\91979\Desktop\Msc Course\Semester 2\Machine Learning\Project 2\ML_dataset\train"
valid_folder = r"C:\Users\91979\Desktop\Msc Course\Semester 2\Machine Learning\Project 2\ML_dataset\valid"
test_folder = r"C:\Users\91979\Desktop\Msc Course\Semester 2\Machine Learning\Project 2\ML_dataset\test"

(train_images, train_annotations, train_objects_counts), (val_images, val_annotations, val_objects_counts), (test_images, test_annotations, test_objects_counts) = load_separate_datasets(train_folder, valid_folder, test_folder)


print(f"Training images: {len(train_images)}, Training annotations: {len(train_annotations)}")
print(f"Validation images: {len(val_images)}, Validation annotations: {len(val_annotations)}")
print(f"Testing images: {len(test_images)}, Testing annotations: {len(test_annotations)}")

# Display object counts by class for each dataset split
print("Training dataset object counts:")
for class_label, count in train_objects_counts.items():
    print(f"{class_label}: {count}")

print("\nValidation dataset object counts:")
for class_label, count in val_objects_counts.items():
    print(f"{class_label}: {count}")

print("\nTesting dataset object counts:")
for class_label, count in test_objects_counts.items():
    print(f"{class_label}: {count}")