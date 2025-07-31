import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from app.data_loader import load_annotations, load_image, SHAPE_ANN_DIR, TRAIN_IMG_DIR, VAL_IMG_DIR

def load_dataset(ann_path, img_dir, n=None):
    df = load_annotations(ann_path, img_dir)
    images = []
    labels = []
    for idx, row in df.iterrows():
        if n and idx >= n:
            break
        image_path = row['image_path']
        if os.path.exists(image_path):
            try:
                images.append(load_image(image_path))
                labels.append(str(row['category']))
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
    return np.array(images), np.array(labels)

def main():
    train_ann = os.path.join(SHAPE_ANN_DIR, 'train_ann_file.txt')
    val_ann = os.path.join(SHAPE_ANN_DIR, 'val_ann_file.txt')

    print("Loading data...")
    X_train, y_train = load_dataset(train_ann, TRAIN_IMG_DIR, n=2000)  # Use n=None for all, or increase as needed
    X_val, y_val = load_dataset(val_ann, VAL_IMG_DIR, n=500)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_train_oh = to_categorical(y_train_enc)
    y_val_oh = to_categorical(y_val_enc)

    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)

    print(f"Classes: {list(le.classes_)}")

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(len(le.classes_), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train_oh, validation_data=(X_val, y_val_oh),
              epochs=12, batch_size=32,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    os.makedirs('./models', exist_ok=True)
    model.save('models/category_model.h5')
    np.save('models/category_classes.npy', le.classes_)
    print("Saved model and class labels.")

if __name__ == "__main__":
    main()
