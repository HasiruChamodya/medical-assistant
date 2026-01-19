from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Define dataset paths
train_dir = r"F:\Project\Medical Assistant\brain_tumor_dataset\Training"
test_dir = r"F:\Project\Medical Assistant\brain_tumor_dataset\Testing"

img_height, img_width = 128, 128
batch_size = 32

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1,
    fill_mode="nearest"
)

# Only rescale for validation/test data (no augmentation!)
val_datagen = ImageDataGenerator(rescale=1./255)

# Set up generator for training
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    shuffle=True
)

# Set up generator for validation
val_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 🟢 SANITY CHECK: Print info after creating val_gen
print("Validation generator class indices:", val_gen.class_indices)
print("Validation generator total images:", val_gen.samples)
print("Validation generator labels count:", np.bincount(val_gen.classes))
for class_name, idx in val_gen.class_indices.items():
    print(f"Class '{class_name}' ({idx}): {np.sum(val_gen.classes == idx)} images")

# Only run for testing/not when imported
if __name__ == "__main__":
    images, labels = next(train_gen)
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    plt.figure(figsize=(12, 6))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        label_index = labels[i].argmax()
        plt.title(idx_to_class[label_index])
        plt.axis('off')
    plt.tight_layout()
    plt.show()