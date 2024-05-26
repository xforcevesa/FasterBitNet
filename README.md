# FasterBitNet

This is the code for FasterBitNet: A Fast and Efficient Framework for High Performance 1/2-bit Quantization.

## Synopsis

FasterBitNet is a PyTorch-based framework for high performance 1/2-bit quantization of neural networks. It is based on the BitNet architecture, which is a convolutional neural network architecture that uses 1/2-bit quantization to reduce the memory footprint and computation time of the network, while maintaining the accuracy of the network, and achieves high throughput and low latency.

It's on the basis of the [BitNet](https://arxiv.org/abs/1905.09788) architecture, which is a convolutional neural network architecture that uses 1/2-bit quantization.

![BitNet Architecture](bitnet.png)

The old README is also available [here](README_OLD.md).

## Structures

The code is based on the PyTorch framework and can be run on any machine with a CUDA-enabled GPU. The code is divided into the following modules:

1. `bitnet`: This module contains the implementation of the BitNet architecture, from the final basical connected layers to Attention Mechanisms and MoE layers, and also even the implementation of the popular neural network layers such as Transformers, Mamba, LlaMA, and so on.
2. `kernel`: This module contains the implementation of the kernel functions used in the BitNet architecture, to perform the 1/2-bit quantization and dequantization of the network's weights and activations.
3. `tests`: This module contains the implementation of the unit tests for the BitNet architecture and the kernel functions.
4. `bit_attention.py`: This module uses our implementation of the BitAttention mechanism to replace the standard attention mechanism in the BitNet architecture.
5. `bit_ffn.py`: This module uses our implementation of the BitFFN architecture to replace the standard FFN architecture in the BitNet architecture.
6. `bit_mamba.py`: This module uses our implementation of the BitMamba architecture to replace the standard Mamba architecture in the BitNet architecture.
7. `kernel_test.py`: This module contains the implementation of the unit tests for the kernel functions, including the correctness tests and the performance tests.
...

The code is still under development and will be updated frequently.

## Requirements

The code requires the following libraries:

```requirements
python>=3.8
torch>=2.0.1
zetascale
einops
```

and also the CUDA-enabled GPU, with the support of Tensor Cores recommended, and CUDA toolkit version 12.0 or higher, which should be installed on your machine.

## Contacts

If you have any questions or suggestions, please feel free to contact me at ```nomodeset@qq.com```.


