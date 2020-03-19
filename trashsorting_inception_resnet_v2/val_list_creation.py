import os,sys
import csv
import time
import time
import glob


now = time.strftime("%Y-%m-%d",time.localtime(time.time()))
class_names_to_ids = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper':3, 'plastic':4, 'trash':5}

data_dir = './dataset/validation/'
output_path = './dataset/validation/' + now + r'val_list.txt'
fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
fd.close()

filename = output_path
myfile = open(filename)

lines = len(myfile.readlines())
print('There are %d lines in %s' % (lines, output_path))






