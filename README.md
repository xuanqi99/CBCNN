# CBCNN

CBCNN is a TensorFlow/Keras implementation of binarized convolutional neural
networks for MNIST, Fashion-MNIST, and GTSRB classification. Convolutional and
dense layers binarize latent real-valued weights to <code>{-H, +H}</code>
during the forward pass. Hidden activations use <code>{-1, +1}</code>, and
gradients are estimated with a Straight-Through Estimator (STE).

Related publication:

> A Lightweight Binarized Convolutional Neural Network Model for Small
> Memory and Low-Cost Mobile Devices  
> <https://doi.org/10.1155/2023/5870630>

## Implementation Scope

This repository uses TensorFlow floating-point operations to simulate binary
weights and activations for training and algorithm evaluation. Actual model
compression or XNOR/popcount acceleration requires weight bit-packing, custom
operators, or a hardware-specific deployment backend.

## Project Structure

~~~text
CBCNN/
├── mnist_cbcnn/
│   ├── binary_ops.py
│   ├── binary_layers.py
│   └── mnist_cbnn.py
├── fashionmnist_cbcnn/
│   ├── binary_ops.py
│   ├── binary_layers.py
│   ├── fashionmnist_cbcnn.py
│   ├── cbcnn_Flops.py
│   └── mnist_reader.py
├── GTSRB_cbcnn/
│   ├── binary_ops.py
│   ├── binary_layers.py
│   ├── GTSRB_cbcnn.py
│   ├── cbcnn_Flops.py
│   └── util.py
└── requirements.txt
~~~

## Requirements

- Python 3.11
- TensorFlow 2.21
- NumPy
- Matplotlib
- OpenCV
- SciPy

All tested package versions are pinned in
[requirements.txt](requirements.txt).

## Environment Setup

The following PowerShell commands create an isolated environment on drive D
with [uv](https://docs.astral.sh/uv/):

~~~powershell
uv python install 3.11 --install-dir D:\tensorflow-python
uv venv D:\tensorflow-envs\cbcnn --python 3.11 --seed
uv pip install --python D:\tensorflow-envs\cbcnn\Scripts\python.exe -r requirements.txt
& D:\tensorflow-envs\cbcnn\Scripts\Activate.ps1
~~~

Alternatively, install the dependencies in any Python 3.11 virtual
environment:

~~~powershell
python -m pip install -r requirements.txt
~~~

TensorFlow 2.11 and later do not support CUDA GPU execution on native Windows.
Use WSL2 when NVIDIA GPU training is required.

## Datasets

### MNIST and Fashion-MNIST

The datasets are downloaded automatically through <code>tf.keras.datasets</code>
on first use.

### GTSRB

Place the preprocessed pickle files at:

~~~text
GTSRB_cbcnn/traffic-signs-data/train.p
GTSRB_cbcnn/traffic-signs-data/test.p
~~~

Each pickle file must contain <code>features</code> and <code>labels</code>.
Images must have shape <code>32 x 32 x 3</code>, and labels must cover the 43
GTSRB classes.

Datasets, model files, checkpoints, generated plots, and training histories are
excluded from version control.

## Training

Run a one-epoch smoke test first:

~~~powershell
python mnist_cbcnn/mnist_cbnn.py --epochs 1
python fashionmnist_cbcnn/fashionmnist_cbcnn.py --epochs 1
python GTSRB_cbcnn/GTSRB_cbcnn.py --epochs 1
~~~

Example full training commands:

~~~powershell
python mnist_cbcnn/mnist_cbnn.py --epochs 50 --batch-size 50
python fashionmnist_cbcnn/fashionmnist_cbcnn.py --epochs 500 --batch-size 50
python GTSRB_cbcnn/GTSRB_cbcnn.py --epochs 1000 --batch-size 200
~~~

Training uses a validation split to select the best weights. Test evaluation is
performed after training. Complete models are saved in the Keras
<code>.keras</code> format, best weights are saved as
<code>.weights.h5</code>, and training curves are saved as PNG files.

## FLOPs

~~~powershell
python fashionmnist_cbcnn/cbcnn_Flops.py
python GTSRB_cbcnn/cbcnn_Flops.py
~~~

The reported value is the operation count of the TensorFlow floating-point
graph. It does not represent the hardware cost of bit-packed binary inference.

## Reproducibility

The training scripts set NumPy and TensorFlow random seeds. Minor numerical
differences may still occur across TensorFlow versions, CPU instruction sets,
and oneDNN execution paths.
