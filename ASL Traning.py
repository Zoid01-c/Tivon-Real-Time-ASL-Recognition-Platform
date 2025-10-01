import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models, optimizers

# === CONFIG ===
# Update these with your own dataset paths
train_dir = r"PATH/TO/ASL/alphabet_train"   # Example: ./dataset/asl_alphabet_train
test_dir  = r"PATH/TO/ASL/alphabet_test"    # Example: ./dataset/asl_alphabet_test

img_size = (224, 224)
batch_size = 32
epochs = 10
model_save_name = "ASL_1642025_model"  # SavedModel format (folder)

# === DATA AUGMENTATION ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# === TEST DATA ===
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# === BASE MODEL SETUP ===
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze from block_13 onward
for layer in base_model.layers:
    if any(block in layer.name for block in ['block_13', 'block_14', 'block_15', 'block_16']):
        layer.trainable = True
    else:
        layer.trainable = False

# === CLASSIFIER HEAD ===
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Ensure final classifier layers are trainable
for layer in model.layers[-5:]:
    layer.trainable = True

# === COMPILE ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === TRAIN ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# === EVALUATE ===
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# === SAVE MODEL (TensorFlow SavedModel Format) ===
model.save(model_save_name)
print(f"\n📁 Model saved as folder: {model_save_name}")

