# Efficient Porting of Large Neural Networks to Low-powered Devices

Neural networks have become a cornerstone in the field of artificial intelligence (AI) and have significantly impacted various domains ranging from healthcare and finance to transportation and entertainment. As these applications become more complex, the size of the neural network models is also increasing–requiring more memory and computation power to run these models. However, a lot of applications require models to run on low-powered devices. As such, reducing the size of a neural network can be essential for deployment on resource-constrained devices, speeding up inference time, and reducing memory footprint. Some of the most common model size reduction techniques include pruning, quantization, knowledge distillation, network compression, etc.

The lottery ticket hypothesis is a concept in deep learning that suggests that within a dense neural network, there exists a sparse subnetwork (the "winning ticket") that, when trained in isolation, can achieve comparable performance to the original dense network. This hypothesis was introduced by Jonathan Frankle and Michael Carbin in their paper titled "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" published in 2019.

In this project, I would like to propose an efficient way to reduce the size and computation of a neural network that can be ported into low-powered edge devices. The usual iterative pruning is an effective way to shrink a large model. However, it requires a model to be trained multiple times before the desired model can be achieved. The current iterative pruning algorithm does not have a time constraint. I would like to add a time constraint to the existing iterative pruning algorithm that will allow users to receive a shrunk model within a given time. Although it might affect the accuracy of the final model, it is suitable for applications where this compromise can be acceptable.

### Following is the proposed pruning algorithm:

1. Randomly initialize a neural network $f(x; θ_0)$ (where $θ_0 ∼ D_θ$).
2. Run 1 epoch and profile the time $t_e$ required for each iteration.
3. Calculate the number of iteration $j$ using $t_e$ for a given time $T$
4. Set the pruning rate $p$ to achieve $f(x; m \odot 0)$ within $j$ iterations.
5. Train the network for j iterations, arriving at parameters $θ_j$. 
6. Prune $p$% of the parameters in $θ_j$, creating a mask $m$. 
7. Reset the remaining parameters to their values in $θ_0$, creating the winning ticket $f(x; m \odot 0)$.

## Evaluation
For the evaluation of the proposed algorithm, I will initially implement and test it with some popular computer vision models and datasets. I would like to expand it to other domains, such as text and audio, to evaluate the effectiveness of the algorithm. I will test how the algorithm impacts accuracy on different models and datasets from different domains.

## Step 2: Dataset Collection
Initially, the proposed pruning algorithm will be tested using some well-known computer vision datasets and models. As such, I have decided to use the following datasets:

**CIFAR10:** The CIFAR-10 dataset is a widely used benchmark dataset in machine learning and computer vision research. It consists of 60,000 color images, each of size 32x32 pixels, belonging to 10 different classes (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck).

The dataset is split into 50,000 training images and 10,000 test images, with each class having an equal number of images. CIFAR-10 is often used for tasks such as object recognition, image classification, and feature extraction in the field of computer vision. It provides a challenging testbed for developing and evaluating machine learning algorithms due to its relatively small image size and the diversity of its classes.

*Download Link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz*

**CIFAR100:** The CIFAR-100 dataset is another widely used benchmark dataset in machine learning and computer vision research, similar to CIFAR-10 but with more fine-grained labels. It consists of 60,000 color images, also of size 32x32 pixels, but these images are divided into 100 different classes, each containing 600 images.

The 100 classes in CIFAR-100 are grouped into 20 superclasses, each containing five fine-grained classes. Some examples of the fine-grained classes include: apple, mushroom, cloud, dolphin, oak tree, and so on.

Similar to CIFAR-10, the CIFAR-100 dataset is split into 50,000 training images and 10,000 test images.

CIFAR-100 is often used for more challenging classification tasks where finer distinctions between classes are necessary, making it suitable for tasks such as fine-grained object recognition, hierarchical classification, and multi-label classification.

*Download Link: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz*

**Tiny-Imagenet:** The Tiny ImageNet dataset is a smaller version of the original ImageNet dataset, which is one of the most widely used datasets in the field of computer vision. The original ImageNet dataset is a large-scale dataset with millions of labeled images spanning thousands of object categories.

Tiny ImageNet is a subset of the original ImageNet dataset. It contains 200 object categories, each with 500 training images, 50 validation images, and 50 test images. All images are of size 64x64 pixels. While smaller in scale compared to the original ImageNet, Tiny ImageNet still provides a challenging testbed for training and evaluating machine learning algorithms for tasks such as image classification, object detection, and image segmentation.

Tiny ImageNet is often used for research and educational purposes, especially in cases where the computational resources required to work with the full ImageNet dataset are not available.

*Download Link: http://cs231n.stanford.edu/tiny-imagenet-200.zip*




