# Holdem

使用强化学习训练德州扑克的智能体。

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

如果需要做开发，建议使用编辑模式来导入项目：

```bash
# pip
pip install -e PATH_TO_THIS_FOLDER
# pdm
pdm add --dev -e PATH_TO_THIS_FOLDER
```

**请不要将包提交到公开pypi server**。

### 使用

训练

```bash
pdm run train
```

### TODO

- [x] kuhn, leduc, texas游戏环境
- [x] wandb训练可视化
- [x] self play训练
- [x] deepstack监督学习
- [ ] 完善test
- [ ] 改进前端（选board牌、过滤不可能combo）
- [ ] 多人德州