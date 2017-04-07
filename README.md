# CMPT-414-CV-OCR

The dependencies python needs to run this program can be found in the requirements.txt file.

To begin training a model, the train function in train.py will need to be modified to point to the correct image 
directories and output directories.

Then to train a model run:

````
python train.py
````

To predict text in an image, modify the main function in predict.py to point towards
the correct path of the image.

Then to predict the image run:

```
python predict
```

misclassification_analysis.py and the debug methods in preprocess.py were added to obtain intermediate data when doing a prediction.


preprocess.py also contains the normalization methods needed to normalize the training data and any images to be run against the trained model.



model.py contains the model of the convolutional neural network.



