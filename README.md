# RealtimeAgeGenderEstimation
This is Pytorch's version implementation of [SSR-Net](https://github.com/shamangary/SSR-Net) for realtime age and gender estimation demo used [MegaAge-Asian](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/) dataset.  

# Platform
- Pytorch
- Keras (for only gender demo)
- Tensorflow (for only gender demo)


It has been tested on google colaboratory.

# Dependencies
- python 3.7
- OpenCV
- moviepy (for demo)

Pipenv is recommended. 
```sh
pipenv --python 3.7.0
pipenv install --skip-lock
```

# Training and testing
```python
python src/train_SSR-Net.py
```
Please make sure the path for dataset is correct.  
The trained weight is saved at trained_models directory.

# DEMO
```python
python demo/demo_webcam.py --model trained_models/pretrained_model_Adam_L1_LRDecay_weightDecay0.0001_batch32_lr0.001_epoch50_pretrained+90_64x64.pth
```
So far, the weight path is trained used the MegaAge-Asian datasets. 

# Reference
The official SSR-Net is here: https://github.com/shamangary/SSR-Net  
See also the pytorch version we inspired: https://github.com/oukohou/SSR_Net_Pytorch