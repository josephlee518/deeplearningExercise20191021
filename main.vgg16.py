"""
Simple Cats vs Dogs Exercise

- With some Augmentation 
    - tensorflow.keras.preprocessin
- With some different Layers (with tf.keras.applications)
    - https://keras.io/applications/ 링크를 참고해서 작성
"""

"""
코드 참고
https://github.com/bradleypallen/keras-dogs-vs-cats/blob/master/keras-dogs-vs-cats-vgg16-transfer.ipynb
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# 학습 안되도록 전환
for layer in model_vgg16_conv.layers:
    layer.trainable = False

# 모델 정의
input = Input(shape=(150, 150, 3))
output_vgg16_conv = model_vgg16_conv(input)
x = Flatten()(output_vgg16_conv)
x = Dense(64, activation='relu')(x)
x = Dense(2, activation='softmax')(x)
# 이걸 모델로 사용
model = Model(inputs=input, outputs=x)
# 모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])

# 베이스 디렉토리 설정
base_dir = "/tmp/cats_and_dogs_filtered"

# Train Dataset
train_dir = os.path.join(base_dir, "train")
train_cats_dir = os.path.join(train_dir, "cats")
train_dogs_dir = os.path.join(train_dir, "dogs")

# Validation Dataset
validation_dir = os.path.join(base_dir, "validation")
validation_cats_dir = os.path.join(validation_dir, "cats")
validation_dogs_dir = os.path.join(validation_dir, "dogs")

# 텐서플로우 학습을 위한 train 및 validation datagen
## 여기가 제일 중요
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255) # 검증

# 학습 데이터를 집어넣기
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

print(train_generator.image_shape)

# 검증 데이터를 집어넣기
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

MODEL_WEIGHTS_FILE = 'vgg16-xfer-weights.h5'
callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]

history = model.fit_generator(
    train_generator,
    callbacks=callbacks,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    verbose=2)