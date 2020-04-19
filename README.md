# Attention, Learn to Solve Routing Problems!

This repository is a **third-party** implementation of [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm). 

*Note: Offical implementation is [here](https://github.com/wouterkool/attention-learn-to-route).*  
*Note: Only TSP is implemented at the present.*


## Requirements

### Hardware and software environment

A Docker image containing whole reproducible environments is provided. 

To pull the image:

```setup
docker pull wpwei/pytorch:latest
```

It is recommended to use GPU to accelerate the traning process. A CUDA compitabile GPU and drivers are needed to be installed.

### Data generation

Training data is generated on the fly. To generate validation and test data (same as used in the paper) for all problems:

```data generation
docker run -v $PWD:/workspace -w /workspace wpwei/pytorch:latest python generate_data.py --problem all --name validation --seed 4321

docker run -v $PWD:/workspace -w /workspace wpwei/pytorch:latest python generate_data.py --problem all --name test --seed 1234
```


## Training

To train the model(s) in the paper, run this command:

```train
docker run --gpus all -v $PWD:/workspace -w /workspace wpwei/pytorch:latest python train.py --n_node <Number of node in TSP problem, default 20>
```

Corresponding validation set will be used by default for `n_node` 20, 50, and 100. For other `n_node`, please specify the validation set using `--val_set` or a validation set will be generated randomly.

During training, loss and validation cost will be logged with tensorboard in logdir `lightning_logs`. To monitor training process with tensorboard, run

```tensorboard
tensorboard --logdir lightning_logs
```
*Note: needs tensorboard installed.*

After training finished, a checkpoint file containing model weights will be generatd in project root folder.

Refer to `train.py` to see other avaliable args.

## Evaluation

To evaluate trained model on test dataset, run:

```eval
docker run --gpus all -v $PWD:/workspace -w /workspace wpwei/pytorch:latest python eval.py --ckpt_path <checkpoint file path> --test_data <test dataset path>
```

## Pre-trained Models

Pre-trained models are in the `pretrained` folder, with `tsp<n_node>_pretrained.ckpt` file name schema.

To evaluate the pretrained model, e.g. on tsp20 test data, run

```eval pretrain
docker run --gpus all -v $PWD:/workspace -w /workspace wpwei/pytorch:latest python eval.py --ckpt_path pretrained/tsp20_pretrained.ckpt --test_data data/tsp/tsp20_test_seed1234.pkl
```

## Results

Our implementation achieves the following performance on TSP problems using the same test data in the paper:

|      Method                       |    TSP n=20        |      TSP n=50      |    TSP n=100     |
| --------------------------------- |------------------- | ------------------ | ---------------- |
|   LKH3 (reported in paper)        |     3.84 (18s)     |      5.70 (5m)     |     7.76 (21m)   |
|   Reported in paper  (greedy)     |     3.85 (0s)      |      5.80 (2s)     |     8.12 (6s)    |
|   Reported in paper  (sampling)   |     3.84 (5m)      |      5.73 (24m)    |     7.94 (1h)    |
|   This repo (greedy)              |     3.88 (0s)      |      5.76 (1s)     |     8.09 (3s)    |
|   This repo (sampling)            |     3.87 (7m32s)   |      5.71 (20m42s) |     7.91 (1h4m)  |

## Acknowledgements

Thanks to the [official implementation repo](https://github.com/wouterkool/attention-learn-to-route) for the data generation scripts.

## Contributing

Any contribution is welcomed. Please feel free to sent me a pull request or drop an issue.