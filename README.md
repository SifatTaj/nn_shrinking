# Efficient Porting of Large Neural Networks to Low-powered Devices

Neural networks have become a cornerstone in the field of artificial intelligence (AI) and have significantly impacted various domains ranging from healthcare and finance to transportation and entertainment. As these applications become more complex, the size of the neural network models is also increasing–requiring more memory and computation power to run these models. However, a lot of applications require models to run on low-powered devices. As such, reducing the size of a neural network can be essential for deployment on resource-constrained devices, speeding up inference time, and reducing memory footprint. Some of the most common model size reduction techniques include pruning, quantization, knowledge distillation, network compression, etc.

In this project, I would like to propose an efficient way to reduce the size and computation of a neural network that can be ported into low-powered edge devices. The usual iterative pruning is an effective way to shrink a large model. However, it requires a model to be trained multiple times before the desired model can be achieved. The current iterative pruning algorithm does not have a time constraint. I would like to add a time constraint to the existing iterative pruning algorithm that will allow users to receive a shrunk model within a given time. Although it might affect the accuracy of the final model, it is suitable for applications where this compromise can be acceptable.

### Following is the proposed pruning algorithm:

1. Randomly initialize a neural network $f(x; θ0)$ (where θ0 ∼ Dθ).
2. Train the network for j iterations, arriving at parameters θj . 
3. Prune p% of the parameters in θj , creating a mask m. 
4. Reset the remaining parameters to their values in θ0, creating the winning ticket f(x; mθ0).
