# LANENET 

**paper**

"LaneNet: Real-Time Lane Detection Networks for Autonomous Driving"


[https://arxiv.org/abs/1807.01726]

**segmentation model [ lane detection ]**
- using pytorch, onnx, TensorRT

## DataSet : TUSimple

dataset URL : [https://github.com/TuSimple/tusimple-benchmark/issues/3]
- unzip train_set.zip

```
|--dataset
|----clips
|----label_data_0313.json
|----label_data_0531.json
|----label_data_0601.json
|----test_label.json
```
## requirement ##
- bash

```shell
 ./requirement.sh
```

## train Lanenet

- lanenet + backbone[ENET]


**how to train**
> 1. TUSimple transform
> ```shell
> python3 tusimple_transform.py --data_set [unzip_tusimple]
> ```
> 2. train Lanenet
> ```shell
> python3 train.py --dataset [TUSimple_dataset(transform)]
> ```
> 3. check save directory [model.pth]
> ```shell
> cd ./save
> ```
>

**how to eval**
> 1. eval Lanenet
> ```shell
> python3 evel.py --dataset [TUSimple_dataset(transform)]
> ```
> 2. check result eval in prompt
>

**how to convert pth to onnx**
> 1. torch to onnx
> ```shell
> python3 pthtoonnx.py --torch_dir ./save/model.pth --onnx_dir ./save
> ```
>
> 2. check onnx model
> ```shell
> ls ./save | grep Lanenet.onnx
> ```
> 

**how to convert onnx to tensorRT**
> 1. onnx to tensorRT
> ```shell
> python3 onnx2trt.py --onnx_dir ./save/Lanenet.onnx --trt_dir ./save
> ```
> 2. check tensorRT
> ```shell
> ls ./save | grep /Lanenet.trt
> ```

## inference model
> **Pytorch inference**
> ```shell
> python3 torch_infer.py --video [mp4] --model ./save/model.pth
> ```
>
>
> **TensorRT inference**
> ```shell
> python3 tensorRT_infer.py --video [mp4] --model ./save/Lanenet.trt
> ```

## inference result

**inference result(video)**

<img width="{100%}" src="https://user-images.githubusercontent.com/67589849/147845566-3577e219-feaa-450d-be32-b9a60d99c84f.gif"/>

video = highway condiction
real velocity = 90 km
h = 256
w = 512

cuda 11.0
onnx 1.6.0
pycuda 2019.1.2
tensorrt 7.1.2.8

hardware : RTX 2080Ti

- Pytorch : 66 fps

- TensorRT : 330 fps

**recommend : use docker**

example for docker 

docker run -it --gpus all -d -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY --name trt20.06 nvcr.io/nvidia/tensorrt:20.06-py3 /bin/bash
