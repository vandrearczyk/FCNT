# FCNT
Texture segmentation with Fully Convolutional Networks

Implementation of the supervised experiment (Experiment B) from the following paper:

    Texture segmentation with Fully Convolutional Networks
    Vincent Andrearczyk and Paul F. Whelan
    arXiv:1703.05230

The Caffe FCN implementation on Shelhamer's GitHub repository must be installed:
https://github.com/shelhamer/fcn.berkeleyvision.org
Note that the fcn8s caffemodel should be downloaded and that the master directory should be added to the PYTHONPATH
The path to cv2.so can be changed in fcnT/solve.py (sys.path.append('/usr/local/lib/python2.7/site-packages')), for instance for other python versions.


The folders and files in this repository must be added to the master FCN repository (root-fcn) as follows.
The 'prague_normal' folder must be copied into the 'root-fcn/data' directory.
The folder with jpg images must be downloaded and untared from the url provided in 'root-fcn/prague_normal/jpegimages-url'.
The 'fcnT' folder and the python files ('prague_helper.py' and 'prague_layers.py') must be copied into the root directory (root-fcn/)

Once everything is installed and copied, you can run solve.py from the 'root-fcn/fcnT' directory as follows:

    usage: solve.py [-h] [--data data] [--n_iter n_iter]

    Supervised texture segmentation of the Prague normal dataset.
    
    optional arguments:
      -h, --help       show this help message and exit
      --data data      number of the test image in the Prague normal texture
                       segmentation dataset (1..20). Default 1
      --n_iter n_iter  number of training iterations. Default 500

The results are saved in 'root-fcn/fcnT/results'

