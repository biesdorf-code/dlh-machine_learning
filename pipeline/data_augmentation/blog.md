Automated Data Augmentation, Step by Step

[BANNER IMAGE, illustrative.]

A neural network that classifies photos needs many labelled examples. Labelling is slow and expensive. Data augmentation is simple: you change the images you already have, a little, in different ways (see below). A dog photographed from the left is still a dog. A dog in a darker room is still a dog. The label is the same but different variations can be made part of the training dataset when training a model.

Automated means the changes happen in code, inside the data pipeline, before the images reach the model. They are random, so the same picture comes back different on the next epoch. In Python, this is straightforwar: one call per operation in TensorFlow: flip, random crop, rotate, brightness, contrast, hue.

Crops help the most. AlexNet took 224 by 224 patches out of 256 by 256 images, which multiplied their training set by 2048.

You can also measure whether a modification is worth keeping. Train twice, once with it and once without, and compare the validation accuracy. Some augmentations help, some end up reducing the accuracy.

- The label must stay the same after the transformation. For example, a mirrored 7 is not a 7.
- Augment the training set only. Validation and test stay fixed.
- Look at a few augmented images before you train on them. Most bugs are obvious for humans to detect, some secondary automation can be used to address this issue as well.
