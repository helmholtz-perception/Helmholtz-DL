## The Helmholtz Method: *Using Perceptual Compression to Reduce ML Complexity*

Created by [Gerald Friedland](http://www.gerald-friedland.org), [Jingkang Wang](http://wangjk.me), [Ruoxi Jia](https://ruoxijia.github.io/), [Bo Li](http://bli89.web.engr.illinois.edu/) and [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/).

![Helmholtz Reinterpretation](https://github.com/helmholtz-perception/Helmholtz-DL/blob/master/docs/Helmholtz.png)

### Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/). We proposed a fundamental answer to a frequently asked question in multimedia computing and machine learning: Do artifacts from perceptual compression contribute to error in the machine learning process and if so, how much? You can also check our [project webpage](http://helmholtz-perception.github.io/Helmholtz-DL/) for a deeper introduction.


Our approach to the problem is a reinterpretation of the Helmholtz Free Energy formula from physics to explain the relationship between content and noise when using sensors (such as cameras or microphones) to capture multimedia data. Extensive experiments show that, at the right quality level, perceptual compression is actually not harmful but contributes to a significant reduction of complexity of the machine learning process. Our work provides insights into the reasons for the success of deep learning.

In this repository, we release code and data for conducting perceptual compression while maintaining, or sometimes even improving, overall performance. Additionally, compressed models usually result in faster convergent speed. 

### Requirements

* **Tensorflow 1.4.0**
* **Keras 2.1.5**
* Python 2.7
* CUDA 8.0+ (For GPU)
* Python Libraries: numpy pandas, h5py, pillow, imageio, librosa and librosa
* [lame](http://lame.sourceforge.net/) tool (for MP3 compression)

The code has been tested with Python 2.7, Tensorflow 1.4.0, CUDA 8.0 and cuDNN 5.1 on Ubuntu 14.04. But it may work on more machines (directly or through mini-modification), pull-requests or test report are well welcomed.

### Usage
To train a model to classify images in CIFAR-10 (quality 5, architecure A)
```
cd cifar
python train.py --quality 5 --setting 0
```
Log files and network parameters will be saved to `logs` folder in default.

To see HELP for the training script:
```
cd cifar && python train.py -h (CIFAR-10)
cd audio && python train.py -h (Audio)
```

If enough GPUs are available, you could use scripts to train models with different settings (architecture, compression ratio). Remember to manually specify proper GPUs in the script (Each process occupies around 3500M graphic memory).
```
cd cifar
sh scripts/train-all-cnns.sh
```

More details are provided in our [project webpage](http://helmholtz-perception.github.io/Helmholtz-DL/).
### Architectures
To evaluate our idea, we have designed six different architectures on CIFAR-10 and Audio dataset, respectively. The details of models (architectures, number of parameters) could be obtained from [cifar_paras](https://github.com/helmholtz-perception/Helmholtz-DL/tree/master/cifar/parameters) and [audio_paras](https://github.com/helmholtz-perception/Helmholtz-DL/tree/master/audio/model/parameters). 

### Citation
If you find our work useful in your research, please consider citing:

	@article{helmholtz18,
	  title={The Helmholtz Method: Using Perceptual Compression to Reduce Machine Learning Complexity},
	  author={Gerald Friedland and Jingkang Wang and Ruoxi Jia and Bo Li and Dawn Song},
	  journal={arXiv preprint arXiv:1804.xxxxx},
	  year={2018}
	}
   
### Acknowledgements
This code is based on the previous works ([All-Conv-Keras](https://github.com/MateLabs/All-Conv-Keras)，[panotti](https://github.com/drscotthawley/panotti)). Many thanks to the authors.

### License
Our code is released under MIT License.
