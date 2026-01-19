import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import train_gen, val_gen

NUM_CLASSES = len(train_gen.class_indices)
IMG_SIZE = (128, 128)

if __name__ == "__main__":
    # 1. Load EfficientNetB0 base model, exclude top
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False  # Freeze all layers to start
    
    # 2. Unfreeze last N layers for fine-tuning
    N = 20
    for layer in base_model.layers[-N:]:
        layer.trainable = True

    # 3. Build classification head
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    # 4. Compile with low learning rate (important for fine-tuning)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 5. Callbacks
    checkpoint = ModelCheckpoint(
        "best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True, verbose=1
    )
    callbacks = [checkpoint, early_stop]

    # 6. Training
    EPOCHS = 15
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # 7. Plot training curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend(), plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(), plt.title("Loss")
    plt.show()

    # 8. Model evaluation
    val_gen.reset()
    pred_probs = model.predict(val_gen)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = val_gen.classes
    cm = confusion_matrix(true_classes, pred_classes)
    print(classification_report(true_classes, pred_classes, target_names=list(val_gen.class_indices.keys())))

    # 9. Save final model (optional, since best_model.h5 is already saved)
    model.save("final_model.h5")