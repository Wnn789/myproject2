import os,sys
import csv
import time
import time
import glob
import random

now = time.strftime("%Y-%m-%d",time.localtime(time.time()))
class_names_to_ids = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper':3, 'plastic':4, 'trash':5}

data_dir = './dataset/training/'
output_path = './dataset/training/' + now + r'data_list.txt'
fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
fd.close()

filename = output_path
myfile = open(filename)
lines = len(myfile.readlines())
print('There are %d lines in %s' % (lines, filename))
myfile.close()


_NUM_VALIDATION = int(0.2*lines)
_RANDOM_SEED = 0
list_path = output_path
train_list_path = './dataset/training/list_train.txt'
val_list_path = './dataset/training/list_val.txt'
fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)
fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:
    fd.write(line)
fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)
fd.close()