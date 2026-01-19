import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
from preprocess import train_gen, val_gen

NUM_CLASSES = len(train_gen.class_indices)
IMG_SIZE = (128, 128)

def print_generator_distribution(generator, steps=3):
    """Prints class distribution from several batches of the generator."""
    print("Sanity check: Checking class label distribution in generator batches...")
    counter = {k: 0 for k in generator.class_indices.keys()}
    batches = 0
    for _, labels in generator:
        label_nums = np.argmax(labels, axis=1)
        for l in label_nums:
            key = list(generator.class_indices.keys())[list(generator.class_indices.values()).index(l)]
            counter[key] += 1
        batches += 1
        if batches >= steps:
            break
    print("Samples per class in generator batches ({} batches):".format(steps), counter)

if __name__ == "__main__":
    # ========== DATA SANITY CHECK ==========
    print("Class indices mapping:", train_gen.class_indices)
    print_generator_distribution(train_gen, steps=5)
    print("Validation class indices mapping:", val_gen.class_indices)
    print_generator_distribution(val_gen, steps=2)

    # ========== MODEL ==========
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = True  # Fine-tune all layers

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ========== CLASS WEIGHTS ==========
    train_labels = train_gen.classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    cw_dict = dict(enumerate(class_weights))
    print("Class weights:", cw_dict)

    # ========== CALLBACKS ==========
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    callbacks = [checkpoint, early_stop, reduce_lr]

    # ========== TRAIN ==========
    history = model.fit(
        train_gen,
        epochs=60,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=cw_dict
    )

    # ========== PLOTS ==========
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend(); plt.title("Accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(); plt.title("Loss")
    plt.show()

    # ========== EVALUATE ==========
    val_gen.reset()
    pred_probs = model.predict(val_gen)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    print(classification_report(true_classes, pred_classes, target_names=class_names))
    cm = confusion_matrix(true_classes, pred_classes)
    print("Confusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    # ========== SAVE ==========
    model.save("final_model.h5")