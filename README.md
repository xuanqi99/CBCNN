# CBCNN

CBCNN 是一个基于 TensorFlow/Keras 的二值卷积神经网络示例项目，包含
MNIST、Fashion-MNIST 和 GTSRB 三个分类任务。卷积层与全连接层在前向传播时
将潜在实值权重二值化为 <code>{-H, +H}</code>，激活函数使用
<code>{-1, +1}</code>，反向传播使用 Straight-Through Estimator（STE）。

项目对应论文：

> A Lightweight Binarized Convolutional Neural Network Model for Small
> Memory and Low-Cost Mobile Devices  
> <https://doi.org/10.1155/2023/5870630>

## 项目状态

代码已适配 Python 3.11、TensorFlow 2.21 和 Keras 3，修复了原始实现中的以下
问题：

- MNIST 卷积与池化层混用 channels_first、channels_last；
- GTSRB 使用测试集统计量预处理训练集造成的数据泄漏；
- 二值层 bias 构建、重复初始化和旧版 Keras API；
- 训练前保存随机模型、旧指标名称及已移除的 fit_generator；
- Fashion-MNIST/GTSRB FLOPs 脚本与 TensorFlow 2 不兼容。

当前实现使用 TensorFlow 浮点卷积模拟二值权重与激活，适合训练和算法验证。
若要获得真实的模型压缩及 XNOR/popcount 加速，还需要额外的权重 bit-pack、
自定义算子或针对目标硬件的部署后端。

## 目录结构

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

## 环境安装

推荐使用 Python 3.11。以下是在 Windows PowerShell 中用
[uv](https://docs.astral.sh/uv/) 创建 D 盘环境的示例：

~~~powershell
uv python install 3.11 --install-dir D:\tensorflow-python
uv venv D:\tensorflow-envs\cbcnn --python 3.11 --seed
uv pip install --python D:\tensorflow-envs\cbcnn\Scripts\python.exe -r requirements.txt
& D:\tensorflow-envs\cbcnn\Scripts\Activate.ps1
~~~

也可以使用任意虚拟环境工具，然后安装：

~~~powershell
python -m pip install -r requirements.txt
~~~

TensorFlow 2.11 及以上版本在原生 Windows 上仅支持 CPU。需要 NVIDIA GPU
训练时，建议在 WSL2 中创建环境。

## 数据集

- MNIST、Fashion-MNIST：首次运行时由 tf.keras.datasets 自动下载。
- GTSRB：将预处理后的 pickle 文件放到
  <code>GTSRB_cbcnn/traffic-signs-data/train.p</code> 和
  <code>GTSRB_cbcnn/traffic-signs-data/test.p</code>。每个文件应包含
  features 与 labels 两个键，图像形状为 32×32×3，类别数为 43。

数据集、模型文件、检查点和训练图片均由 .gitignore 排除，不提交到仓库。

## 训练

先用一个 epoch 检查环境与数据：

~~~powershell
python mnist_cbcnn/mnist_cbnn.py --epochs 1
python fashionmnist_cbcnn/fashionmnist_cbcnn.py --epochs 1
python GTSRB_cbcnn/GTSRB_cbcnn.py --epochs 1
~~~

正式训练可自行设置 epoch 和 batch size：

~~~powershell
python mnist_cbcnn/mnist_cbnn.py --epochs 50 --batch-size 50
python fashionmnist_cbcnn/fashionmnist_cbcnn.py --epochs 500 --batch-size 50
python GTSRB_cbcnn/GTSRB_cbcnn.py --epochs 1000 --batch-size 200
~~~

训练过程使用独立验证集选择最佳权重，完成后才在测试集上评估。模型保存为
Keras .keras 格式，最佳权重保存为 .weights.h5，训练曲线保存为 PNG。

## FLOPs

~~~powershell
python fashionmnist_cbcnn/cbcnn_Flops.py
python GTSRB_cbcnn/cbcnn_Flops.py
~~~

该数值是 TensorFlow 浮点计算图的操作量，不代表 bit-packed 二值推理的硬件
操作量。

## 复现性

代码设置了 NumPy/TensorFlow 随机种子。不同 CPU 指令集、oneDNN 执行顺序和
TensorFlow 版本仍可能造成轻微数值差异。
