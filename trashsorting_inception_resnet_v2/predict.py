from keras.layers import Dense
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

import os
import random
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import decode_predictions
from keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np

def get_train_test_data(list_file):
    list_train = open(list_file)
    x_train = []
    y_train = []
    for line in list_train.readlines():
        x_train.append(line.strip()[:-2])
        y_train.append(int(line.strip()[-1]))
        #print(line.strip())
    return x_train, y_train
x_train, y_train = get_train_test_data('dataset/training/list_train.txt')
x_test, y_test = get_train_test_data('dataset/validation/2020-02-05val_list.txt')
y_test = np.array(y_test) # 测试用：标签数据

def process_train_test_data(x_path):
    images = []
    for image_path in x_path:
        img_load = load_img('dataset/validation/'+image_path)
        img = image.img_to_array(img_load)
        img = preprocess_input(img)
        images.append(img)
    return images
# train_images = process_train_test_data(x_train)
test_images = process_train_test_data(x_test)



# 加载指定模型

base_model = InceptionResNetV2(include_top=False, pooling='avg')
outputs = Dense(6, activation='softmax')(base_model.output)
model = Model(base_model.inputs, outputs)
model.load_weights('train_model/model_68-0.88.hdf5')
# 直接使用predict方法进行预测
y_pred = model.predict(np.array(test_images)[:])
y_pred = np.argmax(y_pred, axis=-1) # 测试用：预测结果数据

print(y_pred[:])
print(y_test[:])

count = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        count += 1

    else:
        pass

print (count)
print('model prediction acc is %10.2f '% (count*100 / len(y_pred)), '%')












