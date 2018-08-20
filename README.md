# ResNet-Tensorflow
Simple Tensorflow implementation of ***pre-activation*** ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
## Summary
### dataset
* [tiny_imagenet](https://tiny-imagenet.herokuapp.com/)
* cifar10, cifar100, mnist, fashion-mnist in `keras` (`pip install keras`)

### Train
* python main.py --phase train --dataset tiny --res_n 18 --lr 0.1

### Test
* python main.py --phase test --dataset tiny --res_n 18 --lr 0.1

## Related works
* [SENet-Tensorflow](https://github.com/taki0112/SENet-Tensorflow)
* [DenseNet-Tensorflow](https://github.com/taki0112/Densenet-Tensorflow)
* [ResNeXt-Tensorflow](https://github.com/taki0112/ResNeXt-Tensorflow)

## Author
Junho Kim
