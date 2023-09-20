

```

python detect.py --config-file configs/PascalVOC-Detection/yolov5-medium.yaml  --source assets/ --weights weights/yolov5-medium-67dd4b3a.pth --data data/voc2012.yaml --data data/voc2007.yaml


python detect.py --config-file configs/PascalVOC-Detection/yolov5-medium.yaml  --source assets/2047 --weights weights/yolov5-m-r1.pth --data data/custom.yaml 

weights/model_best.pth
python detect.py --config-file configs/PascalVOC-Detection/yolov5-medium.yaml  --source assets/2047 --weights weights/model_best.pth --data data/custom.yaml 


python detect.py --config-file configs/PascalVOC-Detection/yolov5-medium.yaml  --source assets/train-2092 --weights weights/model_best.pth --data data/custom.yaml 

```

## train on cutome data

```

python train.py --config-file configs/yolov5-custom.yaml --data data/custom.yaml --epochs 2 --batch-size 2


```


## train on logo data

# singple gpu
```
python train.py --config-file configs/yolov5-custom.yaml --data data/logo-detection-75k-binary/logo-75k-binary.yaml --epochs 2 --device 0


# exp 2 
python train.py --config-file configs/yolov5-custom.yaml --data data/logo-detection-75k-binary/logo-75k-binary.yaml --epochs 50 --device 0

# exp 3
python train.py --config-file configs/yolov5-custom.yaml --data data/logo-75k-binary-image-split/logo-75k-binary.yaml --epochs 50 --device 0

# exp 4
python train.py --config-file configs/yolov5-custom.yaml --data data/logo-75k-binary-image-split/logo-75k-binary.yaml --epochs 50 --device 0 --exp-name exp-4 --batch-size 32

```

# multiple gpu

```
# doesn't work
python -m torch.distributed.launch --nproc_per_node=2 train.py --config-file configs/yolov5-custom.yaml --data data/logo-detection-75k-binary/logo-75k-binary.yaml --epochs 2 --device 0

torchrun --nnodes 1 --nproc_per_node 2 train.py --config-file configs/yolov5-custom.yaml --data data/logo-detection-75k-binary/logo-75k-binary.yaml --epochs 2
```


