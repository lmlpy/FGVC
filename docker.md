## Docker 说明

### 镜像准备
恢复Docker镜像的步骤：
1. 在百度网盘链接下载文件
2. 加载Docker镜像
sudo docker load -i docker.tar
3. 验证镜像
sudo docker images #我们的镜像名为docker

### 数据结构说明
<details>
<summary> 点击可查看data的文件夹目录格式. </summary>

```
data/
├── webfg400/ # WebFG-400 数据集
│ ├── train/ # 训练数据
│ │ ├── 0000/ # 类别 0000
│ │ │ ├── 0001.jpg
│ │ │ └── ...
│ │ └── ... # 其他类别目录
│ └── test_B/ # 测试数据 A
│ ├── 0001.jpg
│ └── ...
├── webfg5000/ # WebFG-5000 数据集
│ ├── train/ # 训练数据
│ │ ├── 0000/ # 类别 0000
│ │ │ ├── 0001.jpg
│ │ │ └── ...
│ │ └── ... # 其他类别目录
│ └── test_B/ # 测试数据 A
│ ├── 0001.jpg
│ └── ...
```
</details>

### bash 命令
```sh
sudo docker run -it --gpus all --shm-size=32g --ipc=host -v xxx/data:/AIC/data docker /bin/bash # xxx为实际数据路径， 数据集结构应该如上(可展开)
#进入docker后应该在/AIC下
bash all.sh
#等待程序运行
#清洗干净的数据会存在/AIC/data/{data_name}/threshold/下 data_name为数据集名称
#训练输出结果在/AIC/log/outputs，包括部分超参数，模型权重文件，标签映射文件，loss和acc的日志文件
#预测输出结果在/AIC/log/results文件夹下的两个csv文件中
```
### 注意
1.需要复现的服务器上已经安装docker、显卡驱动和对应的NVIDIA Container Toolkit \
2.保证允许docker容器有权限修改挂载的data文件夹，因为会将清洗好的数据和划分好的train和val数据存于对应目录 \
3.数据清洗的模型可以直接从国内阿里提供的开放社区modelscope下载，考虑的权重太大所以没有预先下载权重文件哈，需要联网（国内下载很快）


## 提交文件说明

### 一级目录
```bash
代码模型-网络监督细粒度图像识别
├── best_model_web400.pth
├── best_model_web5000.pth
├── docker.md
├── docker.tar
│
├── 源代码
└── 预测结果
```

### 详细目录
```bash
代码模型-网络监督细粒度图像识别
|   技术方案.pdf
|   best_model_web400.pth
|   best_model_web5000.pth
|   docker.md
|   docker.tar
|
+---源代码
|   |   all.sh
|   |   data_cleaner.py
|   |   data_splite.py
|   |   debug.sh
|   |   README.md
|   |   requirements.txt
|   |   train_test.sh
|   |
|   +---code
|   |       datasets.py
|   |       dinov.py
|   |       get_weight.py
|   |       models.py
|   |       models_dinov2.py
|   |       predict.py
|   |       train.py
|   |       utils.py
|   |
|   +---log
|   |   +---my_cache
|   |   +---outputs
|   |   |   +---webfg400
|   |   |   |       args.json
|   |   |   |       class_mapping.json
|   |   |   |       training_history.json
|   |   |   |
|   |   |   \---webfg5000
|   |   |           args.json
|   |   |           class_mapping.json
|   |   |           training_history.json
|   |   |
|   |   \---results
|   |           pred_results_web400.csv
|   |           pred_results_web5000.csv
|   |
|   \---weight
\---预测结果
        pred_results_web400.csv
        pred_results_web5000.cs
```
