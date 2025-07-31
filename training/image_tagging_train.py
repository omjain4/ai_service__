# training/image_tagging_train.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def build_img_model(num_classes):
    base = MobileNetV2(include_top=False, input_shape=(224,224,3), weights='imagenet')
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=outputs)
    return model

model = build_img_model(num_classes=10)  # Update for your number of categories
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Replace the next line with real data!
# model.fit(train_imgs, train_labels, epochs=5, validation_split=0.2)
model.save('../models/clothing_classifier.h5')
