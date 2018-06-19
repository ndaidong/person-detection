# person-detection
To train model for person detection using TensorFlow Object Detection


# Preparation

### Setup workspace

```
# create workspace folder
mkdir workspace
cd workspace

# create virtual environment
python3 -m venv cv-venv
source cv-venv/bin/activate
# install tensorflow or tensorflow-gpu
pip install tensorflow-gpu

# get source code
git clone git@gitlab.com:ggml/person-detection.git

cd person-detection
pip install -r requirements.txt
./init.py

cd tflib
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ..
python tflib/object_detection/builders/model_builder_test.py

```

You have to export PYTHONPATH everytime you start the terminal tab, so it's better to add this to .bash_aliases:


```
echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bash_aliases
```


For more info:

- [TensorFlow Object Detection - Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- [Setup a lightweight environment for deep learning](https://medium.com/@ndaidong/setup-a-simple-environment-for-deep-learning-dc05c81c4914)


### Download dataset


Person dataset:

- http://49.156.52.21:7777/dataset/person/


```
# cd workspace
wget http://49.156.52.21:7777/dataset/person/person-full.tar.gz
tar person-full.tar.gz -zxvf -C person-detection/temp

# now we got the `person-full` directory, check it:
ls person-detection/temp/person-full
```


### Generate TFRecord files


```
# cd workspace/person-detection
./make_tfrecord.py -d temp/person-full -e 10000 -o temp/data -l configs/label_map.pbtxt

```

Arguments:

- `-d`, `--dir`: relative path to folder where we store labels and images
- `-e`, `--extract`: how many images we want to extract from the whole set. Default: `100`.
- `-o`, `--output`: relative path to folder where TFRecord files will be saved into. Default: `temp/data`
- `-l`, `--labelmap`: relative path to label map file. Default `configs/label_map.pbtxt`
- `-r`, `--ratio`: ratio of test set / training set. Default: `0.1` (1 image for test, 9 images for training)

Useful links:

- [TensorFlow Object Detection - Preparing Inputs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)


### Get checkpoints

List of checkpoints:

- http://49.156.52.21:7777/checkpoints/


```
# cd workspace
wget http://49.156.52.21:7777/checkpoints/pretrained/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
tar -zxvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz -C person-detection/temp/checkpoints
```


To find more pretrained models:

- [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)



# Pipeline config


Depending on the selected checkpoint, we use the diffrent pipeline config.


A long list of sample configs are collected here:

- [TensorFlow - Sample config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)


Just download the appropriate config, then specify the following properties:

##### train_config -> fine_tune_checkpoint

Path to pretrained model, e.g `temp/checkpoints/ssd_mobilenet_v2_coco_2018_03_29/model.cpkt`


##### train_input_reader --> tf_record_input_reader --> input_path

Path to `train.record`, e.g `temp/data/train.record`

##### train_input_reader --> label_map_path

Path to label map, e.g `configs/label_map.pbtxt`


##### eval_input_reader --> tf_record_input_reader --> input_path

Path to `eval.record`, e.g `temp/data/eval.record`

##### eval_input_reader --> label_map_path

Path to label map, e.g `configs/label_map.pbtxt`


More about pipeline here:


- [Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)



# Training


```
# cd workspace/person-detection
python tflib/object_detection/train.py --logtostderr --pipeline_config_path=configs/PATH_TO_PIPELINE.config --train_dir=PATH_TO_TRAIN_DIR
```

`PATH_TO_TRAIN_DIR` is where we would to save trained model into. It will be created automatically if not yet.

Recommended path: `temp/models/MODEL_NAME/MODEL_VERSION/train`


### Evaluation


```
python tflib/object_detection/eval.py --logtostderr --pipeline_config_path=PATH_TO_PIPELINE.config --checkpoint_dir=PATH_TO_TRAINING_DIR --eval_dir=PATH_TO_EVALUATION_DIR

```

`PATH_TO_TRAIN_DIR`: training model directory that we've specified above.

`PATH_TO_EVALUATION_DIR`: path to where to store evaluating result. It will be created automatically if not yet.


Recommended path: `temp/models/MODEL_NAME/MODEL_VERSION/eval`


PATH_TO_PIPELINE_CONFIG


### TensorBoard

```
tensorboard --logdir=training:PATH_TO_TRAIN_DIR,test:PATH_TO_EVALUATION_DIR
```

Useful links:

- [TensorFlow Object Detection - Running Locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)


# Export graph


```
python tflib/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path=PATH_TO_PIPELINE.config --trained_checkpoint_prefix=PATH_TO_TRAINING_DIR/model.ckpt-INDEX --output_directory=PATH_TO_OUTPUT_DIR
```

Useful links:

- [Exporting a trained model for inference](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md)
- “[How to deploy an Object Detection Model with TensorFlow serving](https://medium.freecodecamp.org/how-to-deploy-an-object-detection-model-with-tensorflow-serving-d6436e65d1d9)” by [@KailaGaurav](https://twitter.com/KailaGaurav)


# Prediction


```
./predict.py -m PATH_TO_FROZEN_INFERENCE_GRAPH.pb -l PATH_TO_LABEL_MAP_FILE.pbtxt

```

Arguments:

- `-m`, `--model`: relative path to exported `.pb` graph file
- `-l`, `--labelmap`: relative path to label map file. Default `configs/label_map.pbtxt`
- `-f`, `--file`: path to image or video file. Only support `.jpg`, `.png`, `.mp4`, `.avi`, `.mkv`
- `-c`, `--cam`: index of camera. Use this argument for realtime prediction with specified camera.
- `-d`, `--dir`: path to image foldes. Use this argument for predict multi images. Default `samples`
- `-o`, `--output`: relative path to output folder. Go with `--dir`. Default `temp/result`



# Tested with

- TensorFlow v1.6, v1.7, v1.8
- Python v3.6.4, v3.6.5
- Ubuntu 17.10, 18.04


# License

The MIT License (MIT)

