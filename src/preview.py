import os
import matplotlib.pyplot as plt
import cv2

# dataset paths
train_dir = r"F:\Project\Medical Assistant\brain_tumor_dataset\Training"
test_dir = r"F:\Project\Medical Assistant\brain_tumor_dataset\Testing"

# 1. List class folders (for multi-class setup)
train_classes = os.listdir(train_dir)
print("Training class folders:", train_classes)

# 2. Count images per class
for cls in train_classes:
    cls_path = os.path.join(train_dir, cls)
    print(f"{cls}: {len(os.listdir(cls_path))} images")

# 3. Visualize sample images from each class
num_examples = 5
plt.figure(figsize=(8, 6))
for i, cls in enumerate(train_classes):
    cls_path = os.path.join(train_dir, cls)
    img_files = os.listdir(cls_path)[:num_examples]
    for j, img_file in enumerate(img_files):
        img_path = os.path.join(cls_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(len(train_classes), num_examples, i * num_examples + j + 1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')
plt.tight_layout()
plt.show()