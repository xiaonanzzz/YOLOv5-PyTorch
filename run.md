

```

python detect.py --config-file configs/PascalVOC-Detection/yolov5-medium.yaml  --source assets/ --weights weights/yolov5-medium-67dd4b3a.pth --data data/voc2012.yaml --data data/voc2007.yaml




```

## train on cutome data

```

python train.py --config-file configs/yolov5-custom.yaml --data data/custom.yaml --epochs 2 --batch-size 2


```


