from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define dataset paths
train_dir = r"F:\Project\Medical Assistant\brain_tumor_dataset\Training"
test_dir = r"F:\Project\Medical Assistant\brain_tumor_dataset\Testing"

img_height, img_width = 128, 128
batch_size = 32

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values
    rotation_range=15,      # Randomly rotate images
    width_shift_range=0.1,  # Random width shift
    height_shift_range=0.1, # Random height shift
    shear_range=0.1,        # Shear in random direction
    zoom_range=0.1,         # Random zoom
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill in new pixels after a shift/rotate
)

# Only rescale for validation/test data (no augmentation!)
val_datagen = ImageDataGenerator(rescale=1./255)

# Set up generator to read images, assign labels, and augment as needed
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)



# Only run for testing—not when imported
if __name__ == "__main__":
    images, labels = next(train_gen)  # Get one batch of images and labels

    # Reverse label map from index to class name
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