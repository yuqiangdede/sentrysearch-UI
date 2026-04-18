# SentrySearch

SentrySearch 是一个视频语义检索工具。你输入一句自然语言，它会在视频里找出相关片段，并可以直接裁剪成短视频。

## 做什么

- 把视频切成重叠片段
- 使用 Gemini 或本地 Qwen3-VL-Embedding 生成向量
- 把向量存到本地 ChromaDB
- 搜索时把文字转成同一向量空间进行匹配
- 支持自动裁剪结果
- 支持本机 Web 控制台

## 目录

- `sentrysearch/`：核心代码
- `tests/`：测试
- `scripts/`：下载模型脚本
- `models/`：项目内模型文件
- `sample_data/`：示例视频
- `uploads/`：Web 上传目录
- `.venv/`：项目内虚拟环境

## 环境要求

- Python 3.11+
- `ffmpeg`
- Windows 建议使用 PowerShell

## 下载模型

默认本地模型是 `Qwen3-VL-Embedding-2B`，下载后放在项目目录内：

```text
./models/Qwen3-VL-Embedding-2B
```

推荐直接用仓库内脚本下载：

```powershell
.\.venv\Scripts\python.exe .\scripts\download_models.py --model qwen2b
```

如果你还想把 8B 一起下载下来：

```powershell
.\.venv\Scripts\python.exe .\scripts\download_models.py --model all
```

说明：

- 模型文件必须放在项目目录内
- 不要默认依赖项目目录外的模型缓存
- 本项目当前默认只用 2B，不再默认切 8B

## 安装

先在项目根目录创建项目内虚拟环境：

```powershell
uv venv .venv
```

安装基础依赖：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -e .
```

如果你要使用本地模型，再安装本地依赖：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -e ".[local]"
```

如果你要用更省显存的量化版本：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -e ".[local-quantized]"
```

## 启动

### Web 控制台

在项目根目录运行：

```powershell
.\.venv\Scripts\sentrysearch.exe web --host 127.0.0.1 --port 8000
```

或者：

```powershell
.\.venv\Scripts\python.exe -m sentrysearch.cli web --host 127.0.0.1 --port 8000
```

然后在浏览器打开：

```text
http://127.0.0.1:8000
```

### 命令行

查看帮助：

```powershell
.\.venv\Scripts\sentrysearch.exe --help
.\.venv\Scripts\sentrysearch.exe index --help
.\.venv\Scripts\sentrysearch.exe search --help
```

索引示例：

```powershell
.\.venv\Scripts\sentrysearch.exe index .\sample_data --backend local --model .\models\Qwen3-VL-Embedding-2B
```

搜索示例：

```powershell
.\.venv\Scripts\sentrysearch.exe search "前车突然变道" --backend local --model .\models\Qwen3-VL-Embedding-2B
```

如果你要用 Gemini 云端模式，先配置 API Key：

```powershell
.\.venv\Scripts\sentrysearch.exe init
```

然后就可以直接索引和搜索：

```powershell
.\.venv\Scripts\sentrysearch.exe index .\sample_data
.\.venv\Scripts\sentrysearch.exe search "前车突然变道"
```

## Web 控制台怎么用

Web 页面里主要有三步：

1. 上传视频
2. 建立索引
3. 搜索并裁剪

默认建议直接使用本地 2B 模型：

- 目标模型路径：`./models/Qwen3-VL-Embedding-2B`
- 本地索引和搜索都用这个路径

## 常用命令

```powershell
# 查看索引状态
.\.venv\Scripts\sentrysearch.exe stats

# 删除某个视频的索引
.\.venv\Scripts\sentrysearch.exe remove 你的视频文件名

# 清空索引
.\.venv\Scripts\sentrysearch.exe reset
```

## 说明

- 项目内所有模型、资源、上传目录都应放在项目目录下
- `.venv` 也应放在项目根目录
- 本地模型默认使用 `Qwen3-VL-Embedding-2B`
- 如果没有本地模型，就用 Gemini 云端模式

## 许可证

MIT 许可证
