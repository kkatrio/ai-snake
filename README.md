### Overview

Reinforcement learning on snake game: Train a snake to grow using convolutional layers of states.

![run](https://dikatrio.xyz/img/run1.gif)

### Algorithm

I used DQN to train the agent. I used the keras API and TensorFlow's [conv2d](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) convolutional layers to build the neural network. The input is a multi-channel sequence of states. The output is the action-value Q-function, which we try to optimize, as I try to describe in a [blog post](https://dikatrio.xyz/posts/train_snake/).

### Requirements

- Tensorflow 2.1
- Python 3.6
- numpy 1.18
- pytest 5.4

Tensorflow runs on CPU

### Training

To train the snake with the default parameters use 
```
make train
```

Run the unit tests with
```
make tests
```
(pseudo-random number generation involved)

### Visualization

To run an episode with a trained snake model in the data directory, use
```
make run
```

This uses a trained model in the data directory. If you train a snake, you can run an episode using the script directly:
```
./run_evolved_snake.py <tf model>
```
