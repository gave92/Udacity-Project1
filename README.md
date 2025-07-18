# Object detection in an urban environment

Objective: training an object detection model using the Tensorflow Object Detection API, AWS Sagemaker and transfer learning on pre-trained NN of the Tensorflow object detection model zoo.

<p align="center">
    <img src="data/output_rcnn.gif" alt="drawing" width="600"/>
</p>

## Data

Train data is from camera images of the Waymo Open Dataset. As a first step the images with groundtruth from the training data were download (in .tfrecord format) to get an idea of what the training data contained. A few of the images were displayed using matplotlib and a statistic of the groundtruth classes across the train data was calculated. The statistic shows a vast majority of the dataset contains vehicles, less pedastrians and very few cyclists.

| Vehicles | Pedestrians | Cyclists |
| -------- | ------- | ------- |
| 77%  |  22%   | 1%    |

## Model testing

The model tested for this project were:

| Model | Config file |
| -------- | ------- |
| EfficientDet D1  |  [pipeline.config](https://github.com/gave92/Udacity-Project1/blob/29985991397fd4fbfe87ed3635c65e79716f2e52/1_model_training/source_dir/pipeline.config)  |
| SSD with Mobilenet v2 FPN-lite  |  [pipeline_mobile.config](https://github.com/gave92/Udacity-Project1/blob/29985991397fd4fbfe87ed3635c65e79716f2e52/1_model_training/source_dir/pipeline_mobile.config)   |
| Faster R-CNN with Resnet-50 (v1)  |  [pipeline_rcnn.config](https://github.com/gave92/Udacity-Project1/blob/29985991397fd4fbfe87ed3635c65e79716f2e52/1_model_training/source_dir/pipeline_rcnn.config)   |

## Data augmentation options


[displayDataAugmentations.py](https://github.com/gave92/Udacity-Project1/blob/29985991397fd4fbfe87ed3635c65e79716f2e52/scripts/displayDataAugmentations.py)
