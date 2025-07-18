import tensorflow as tf
from utils import get_dataset

dataset = get_dataset("./train/*.tfrecord","./train/label_map.pbtxt")

classes = {1:0,2:0,4:0}
for data in dataset.take(10000):
    for cl in data['groundtruth_classes'].numpy():
        classes[cl] += 1
print("Cars: {:.0%}".format(classes[1]/(sum(classes.values()))))
print("Pedestrians: {:.0%}".format(classes[2]/(sum(classes.values()))))
print("Cyclists: {:.0%}".format(classes[4]/(sum(classes.values()))))
