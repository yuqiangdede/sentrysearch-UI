# 项目内路径约定

SentrySearch 现在把运行数据默认收敛到项目根目录下：

- 向量数据库：`./.sentrysearch/db`
- 运行时临时文件：`./.sentrysearch/tmp`
- 配置文件：`./.sentrysearch/.env`
- 本地模型：`./models`
- 上传视频：`./uploads/videos`
- 裁剪输出：`./clips_output`
- 虚拟环境：`./.venv`

说明：
- 这些目录都在项目目录内
- 索引、裁剪和上传默认不会写到用户目录
- 如果某个路径参数传入项目外路径，程序会报错并拒绝写入
