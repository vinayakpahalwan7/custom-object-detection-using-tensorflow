# custom-object-detection-using-tensorflow
 Track and detect our own custom objects with Tensorflow

Brief Overview:

1. Collect a few hundred images that contain your object - ideal is 500+.
2. Annotate/label the images, ideally with a program LabelImg/RectLabel - to create an XML file.
3. Split this data into train/test samples.
4. Generate TF Records from these splits.
5. Setup a .config file for the model of choice.
6. Train.
7. Export graph.
8. Detect custom objects.

For Annotations: 
RectLabel (https://itunes.apple.com/us/app/rectlabel-for-object-detection/id1210181730?mt=12)
LabelImg (https://github.com/tzutalin/labelImg)
Once you have over 100 images labeled, we're going to separate them into training and testing groups. To do this, just copy about some 15% of your images and their annotation XML files to a new dir called 'test' and then copy the remaining ones to a new dir called 'train'

Create a new folders 'data' and 'training' & first convert XML files to csv using ```xml_to_csv.py```

Generate Tfrecords by using ```generate_tfrecord.py```

Goto : git clone https://github.com/tensorflow/models.git and follow install isntructions.

Run generate_tfrecord.py
```
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
```
```
python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
```

Now, in your data directory, you should have train.record and test.record.

Use pre-trained model for faster training.

Download checkpoint and config files:
```
wget https://raw.githubusercontent.com/tensorflow/models/master/object_detection/samples/configs/ssd_mobilenet_v1_pets.config
```

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
```

Extract ssd_mobilenet to main directory

Modify 'PATH_TO_BE_CONFIGURED' and batch size in the config file (else it would give you a memory error).

Inside 'training' dir create a '.pbtxt' file.

Train:
```
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

Launch tensorboard: 
```
tensorboard --logdir='training' (127.0.0.1:6006)
```

Export Inference graph : 
```
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-10856 \
    --output_directory apple_inference_graph
```

Check for 'pipeline_config_path' in the config file (set appropriate path)

Open JUPYTER NOTEBOOK : Click on object_detection_tutorial.ipynb and change variables 'MODEL_NAME', 'PATH_TO_CKPT', 'PATH_TO_LABELS', 'NUM_CLASSES' & 'TEST_IMAGE_PATHS'

Example:
```
MODEL_NAME = 'apple_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'apple.pbtxt')
NUM_CLASSES = 1
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(2, 6) ]
```

Click on 'Cell' menu and 'Run All'.
