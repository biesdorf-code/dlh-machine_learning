ImageNet Classification with Deep Convolutional Neural Networks

Krizhevsky, Sutskever and Hinton, NIPS 2012. A summary.

Image from the PDF, the architecture.

Introduction

Until 2012, object recognition used hand-designed features and small image collections of tens of thousands of pictures. Real photos vary too much for that. ImageNet changed the situation: 15 million labelled images in 22,000 categories. The authors wanted to demonstrate that a large convolutional neural network, trained directly on those images, could learn the features itself and beat every previous method.

Procedures

The network has eight learned layers: five convolutional and three fully connected, 60 million parameters in total. It was trained on 1.2 million ImageNet pictures for 1,000 classes, using two GTX 580 GPUs of 3 GB each, for about six days. Three ideas made it possible. ReLU units instead of tanh, which train much faster. Data augmentation with random crops and colour variation. And dropout in the fully connected layers.

Results

On ILSVRC-2010 the network obtained 37.5% top-1 and 17.0% top-5 error, against 47.1% and 28.2% for the previous best. In the 2012 competition it reached 15.3% top-5 error, while the second place obtained 26.2%. Depth was essential: removing a single convolutional layer cost about 2% accuracy. The learned filters recognise edges, colours and blobs, and similar images produce similar internal vectors.

Conclusion

The authors conclude that a large, deep convolutional network trained with supervision alone can solve a very difficult recognition problem, and that its depth is what makes it work. No unsupervised pre-training was necessary. Their results were still limited by available GPU memory and training time, so they expected better numbers simply from bigger networks, and they proposed to apply the same approach to video.

Personal Notes

Using Dropout was used to reduce overfitting, it not only made the model more robustbut also doubled the time for the training to converge. It has it's costs, but it is viable because nowadays we have much more compute power. Splitting the work across infra is a concern of ML engineers, not only ML Ops Engineers.