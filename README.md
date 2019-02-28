# CIFAR-10


The advantage of multiple layers is that they can learn features at various levels of abstraction. For example, if you train a deep CNN to classify images, you will find that the first layer will train itself to recognize very basic things like edges, the next layer will train itself to recognize collections of edges such as shapes, the next layer will train itself to recognize collections of shapes like wheels, legs, tails, faces and the next layer will learn even higher-order features like objects (truck, ships, dog, frog etc). Multiple layers are much better at generalizing because they learn all the intermediate features between the raw input and the high-level classification. At the same time, there are few important aspects which need to be taken care off to prevent over-fitting. Deep CNN are harder to train because:

a)  Data requirement increases as the network becomes deeper.
b) Regularization becomes important as number of parameters (weights) increases in order to do learning of weights from memorization of features towards generalization of features.