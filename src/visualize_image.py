import matplotlib.pyplot as plt
import cv2

num_examples = 2

plt.figure(figsize=(8, 6))
for i, cls in enumerate(train_classes):
    cls_path = os.path.join(train_dir, cls)
    img_files = os.listdir(cls_path)[:num_examples]
    for j, img_file in enumerate(img_files):
        img_path = os.path.join(cls_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert for matplotlib
        plt.subplot(len(train_classes), num_examples, i*num_examples + j + 1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')
plt.tight_layout()
plt.show()