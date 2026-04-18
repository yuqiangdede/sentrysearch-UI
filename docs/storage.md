# 存储位置

SentrySearch 的视频向量索引默认保存在项目根目录下：

```text
./.sentrysearch/db
```

说明：
- 这是本地 ChromaDB 的持久化目录
- 不建议手动删除，除非你要清空索引
- 该目录已加入 `.gitignore`
