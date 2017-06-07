# Lung Segmentation (2D)
Repository features [UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) inspired architecture used for segmenting lungs on chest X-Ray images.

## Demo
See the application of the model in [Demo](https://github.com/imlab-uiip/lung-segmentation-2d/tree/master/Demo) folder.

## Implementation
Implemented in Keras(2.0.4) with TensorFlow(1.1.0) as backend. 

Use of data augmentation for training required slight changes to keras ImageDataGenerator. Generator in `image_gen.py` applies same transformation to both the image and the label mask.

To use this implementation one needs to load and preprocess data (see `load_data.py`), train new model if needed (`train_model.py`) and use the model for generating lung masks (`inference.py`).

`trained_model.hdf5` contains model trained on both data sets mentioned below.

## Segmentation
Scores achieved on [Montgomery](https://openi.nlm.nih.gov/faq.php#faq-tb-coll) and [JSRT](http://www.jsrt.or.jp/jsrt-db/eng.php)(With [these masks](http://www.isi.uu.nl/Research/Databases/SCR/). See `preprocess_JSRT.py`.) (Measured using 5-fold cross-validation):

|      |  JSRT | Montgomery |
|:----:|:-----:|:----------:|
|  IoU | 0.971 |    0.956   |
| Dice | 0.985 |    0.972   |

![](http://imgur.com/BAAvFnp.png) ![](http://imgur.com/uQYW7Da.png)

![](http://imgur.com/jOVJFtD.png) ![](http://imgur.com/N2AM9PL.png)

