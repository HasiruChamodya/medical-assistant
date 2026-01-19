from preprocess import train_gen, val_gen

# Fetch one batch of images and labels
images, labels = next(train_gen)

print(f"Images shape: {images.shape}")      # Should be (batch_size, height, width, channels)
print(f"Labels shape: {labels.shape}")      # Should be (batch_size, num_classes)
print("Class indices:", train_gen.class_indices)  # Shows folder->label mapping
print("First image label:", labels[0])      # Shows the one-hot label for the first sample

# Check class distribution in this batch
print("Label sums, first batch (should see a spread across classes):", labels.sum(axis=0))