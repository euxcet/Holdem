# AlphaHoldem Pytroch

尝试将AlphaHoldem移植到最新的pytorch版本下。

## 部署

### PDM

项目采用[PDM](https://github.com/pdm-project/pdm)构建，这是个类似于poetry的包管理器。

**Linux/Mac 安装命令**

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

```bash
brew install pdm
```

```bash
pip install --user pdm
```

**Windows 安装命令**

```powershell
(Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing).Content | python -
```

### 安装

本项目Python 3.11 或更高版本，可使用conda创建环境：

```bash
conda create -n [YOUR_ENV_NAME] python=3.11
```

在虚拟环境中安装本库：

```bash
pdm build
pdm install
```

在构建时需要在虚拟环境中安装包，可使用国内镜像：

```bash
pdm config pypi.url https://pypi.tuna.tsinghua.edu.cn/simple/
```

在dist文件夹中会生成whl文件，可在其他环境中使用pip安装：

```bash
pip install --force-reinstall dist/alphaholdem-*-py3-none-any.whl
```

**请不要将包提交到公开pypi server**。

### 使用

进行一次随机的游戏：

```bash
pdm run game
```

### TODO

- [ ] 实现双人翻前All-in or Fold策略训练
- [ ] wandb可视化
- [ ] 不同self-play算法
- [ ] Trinal-Clip loss