# manifest 文件生成与更新策略 PRD

适用路径：`D:\code3\SecondBrain\data\manifest`

## 1. 文件清单（按路径排序）
1) `data/manifest/kb_paths.json`  
2) `data/manifest/<kb_id>/files.json`  
3) `data/manifest/<kb_id>/index_manifest.json`  
4) `data/manifest/<kb_id>/ingest.log`  
5) `data/manifest/<kb_id>/ingest_report.json`

---

## 2. 各文件生成与更新策略

### 2.1 `data/manifest/kb_paths.json`
**用途**：记录“根目录绝对路径 → kb_id”的映射。  
**生成策略**：
- 第一次对某个根目录执行 ingest 时创建或更新。
- 未指定 `--kb` 时，自动生成 kb_id（基于目录名 + hash 后缀）。

**更新策略**：
- 当同一路径再次 ingest 且传入 `--kb`，会覆盖该路径对应的 kb_id。
- 新增路径会追加记录，不删除历史路径记录。

---

### 2.2 `data/manifest/<kb_id>/files.json`
**用途**：知识库内“文件状态清单”。  
**生成策略**：
- 每次 ingest 结束都会写入（全量覆盖）。
- 由 ingest 过程扫描根目录生成。

**更新策略**：
- 对每个文件记录以下字段：`abs_path`、`rel_path`、`hash`、`mtime_ns`、`status`、`source_deleted`、`error`、`updated_at`。
- 已入库且文件未变化：保持 `status=indexed`，仅在需要时更新 `updated_at`。
- 文件被删除：标记 `source_deleted=true` 且 `status=deleted`。
- 解析/入库失败：`status=error` 并写入 `error`。
- 不支持类型：`status=skipped`，`error=unsupported`。

---

### 2.3 `data/manifest/<kb_id>/index_manifest.json`
**用途**：索引元信息（embedding 模型、chunk 参数、数量、更新时间等）。  
**生成策略**：
- 只有在本次 ingest **确实有新文档入库**（`any_indexed=True`）时生成/更新。

**更新策略**：
- 每次有新增/更新入库时覆盖写入。
- 内容包含：`embedding_model`、`chunk_size`、`chunk_overlap`、`doc_count`、`chunk_count`、`updated_at`。
- 若本次 ingest 无新增入库，不更新该文件。

---

### 2.4 `data/manifest/<kb_id>/ingest.log`
**用途**：入库过程日志（INFO/WARN/ERROR）。  
**生成策略**：
- 每次 ingest 时配置日志输出到该文件（追加写入）。

**更新策略**：
- 日志级别受 `RAG_LOG_LEVEL` 控制。
- 每次 ingest 会追加新的日志段，不会自动清理历史日志。

---

### 2.5 `data/manifest/<kb_id>/ingest_report.json`
**用途**：本次 ingest 汇总报告。  
**生成策略**：
- 每次 ingest 结束都会生成（全量覆盖）。

**更新策略**：
- 覆盖写入“本次 ingest”的摘要：  
  - `ingested`（成功入库）  
  - `skipped`（未变化/不支持/已删除）  
  - `errors`（解析失败/无内容）  
  - `created_at`（本次 ingest 时间）

---

## 3. 一致性说明
- `files.json` 是“文件清单视图”，`index_manifest.json` 是“索引视图”。  
- 在 ingest 中如果索引失败或中断，可能出现 **files.json 更新了但索引未完成** 的短暂不一致。  
- 通过 `ingest_report.json` 可定位本次入库是否有失败项。

---

## 4. 建议的运维操作
- 发现文件列表与检索结果不一致：  
  1) 查看 `ingest_report.json` 是否有 errors  
  2) 查看 `ingest.log` 定位错误原因  
  3) 重新 ingest 对应目录

