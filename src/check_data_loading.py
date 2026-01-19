from preprocess import train_gen, val_gen

images, labels = next(train_gen)

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print("Class indices:", train_gen.class_indices)
print("First image label:", labels[0])