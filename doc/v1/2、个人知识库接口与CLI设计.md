# 个人知识库 接口/CLI 设计

版本：v1.0
日期：2026-01-23

## 1. 目标
提供一致、可脚本化的接口与 CLI，覆盖知识库生命周期（创建、入库、索引、检索、统计），并支持本地优先的使用方式。

## 2. CLI 设计
命令前缀建议：`ragkb`（也可作为 Python 模块入口 `python -m ragkb`）。

### 2.1 全局参数
- `--kb <kb_id>`：指定知识库 ID
- `--root <path>`：项目根目录（默认当前目录）
- `--output json|text`：输出格式（默认 text）
- `--log-level debug|info|warn|error`
- `--dry-run`：仅展示将要执行的动作

### 2.2 知识库管理
#### 2.2.1 创建知识库
```
ragkb kb create --name "项目A" --kb project_a --desc "合同与资料" --tags "合同,采购"
```
输出（json）：
```
{ "kb_id": "project_a", "name": "项目A", "created_at": "..." }
```

#### 2.2.2 列出知识库
```
ragkb kb list
```

#### 2.2.3 查看知识库详情
```
ragkb kb info --kb project_a
```

#### 2.2.4 删除知识库
```
ragkb kb delete --kb project_a --force
```

### 2.3 入库与增量更新
#### 2.3.1 导入文件/目录
```
ragkb ingest add --kb project_a --path "D:\docs" --watch
```
参数：
- `--path`：文件或目录（必须是 Win10/Linux 绝对路径；禁止项目内目录）
- `--watch`：开启目录监听（可选）
- `--tags`：给本次导入打标签
说明：
- 新增“文件”无需改配置；仅新增“源目录”时需要写入路径映射（见 2.8）。

#### 2.3.2 扫描并增量更新
```
ragkb ingest scan --kb project_a --path "D:\docs"
```
行为：
- 新文件 → 解析 + 入库
- 变更文件 → 重新解析 + 更新索引
- 删除文件 → 标记“源文件已删除”，保留索引与解析产物；回答时注明来源已删除

### 2.4 解析与 OCR
#### 2.4.1 强制重新解析
```
ragkb parse rebuild --kb project_a --doc "合同_2025.pdf"
```

### 2.5 索引
#### 2.5.1 构建索引
```
ragkb index build --kb project_a
```

#### 2.5.2 重建索引
```
ragkb index rebuild --kb project_a --force
```

### 2.6 检索与问答
#### 2.6.1 检索
```
ragkb search --kb project_a --query "违约金条款" --top-k 10
```

#### 2.6.2 问答
```
ragkb ask --kb project_a --question "合同里的违约金是多少？" --top-k 6
```
输出：
- answer
- citations（文件名、页码/段落、source_deleted 标记）

### 2.7 统计与诊断
#### 2.7.1 查看统计
```
ragkb stats --kb project_a
```

#### 2.7.2 查看解析失败列表
```
ragkb errors --kb project_a --type ocr
```

#### 2.7.3 检查索引清单
```
ragkb index manifest --kb project_a
```

### 2.8 配置与环境
#### 2.8.1 查看当前配置
```
ragkb config show
```
#### 2.8.2 管理“源路径 → KB”映射
```
ragkb config paths list
ragkb config paths add --kb project_a --path "D:\docs\project_a"
ragkb config paths remove --path "D:\docs\project_a"
```
说明：
- 映射文件建议落在 `data/manifest/kb_paths.json`，用于追溯与一致归属。
- 新增“文件”不需要改配置；新增“源目录”才需要写入映射。

## 3. API 设计（可选，若提供本地服务）
建议提供 FastAPI 服务，CLI 默认直接调用本地逻辑，或通过 `--api` 切换。

### 3.1 基础
- Base URL: `http://localhost:8000`
- Content-Type: `application/json`
- 认证：默认无；可选 `X-API-Key`

### 3.2 接口列表
#### 3.2.1 创建 KB
`POST /kb`

Request:
```
{ "kb_id": "project_a", "name": "项目A", "desc": "合同资料", "tags": ["合同"] }
```
Response:
```
{ "kb_id": "project_a", "created_at": "..." }
```

#### 3.2.2 列出 KB
`GET /kb`

#### 3.2.3 KB 详情
`GET /kb/{kb_id}`

#### 3.2.4 删除 KB
`DELETE /kb/{kb_id}`

#### 3.2.5 导入文件
`POST /kb/{kb_id}/ingest`

Request:
```
{ "path": "D:/docs", "watch": false, "tags": ["合同"] }
```

#### 3.2.6 扫描增量
`POST /kb/{kb_id}/scan`

#### 3.2.7 索引构建/重建
`POST /kb/{kb_id}/index/build`
`POST /kb/{kb_id}/index/rebuild`

#### 3.2.8 检索
`POST /kb/{kb_id}/search`

Request:
```
{ "query": "违约金", "top_k": 10, "filters": {"tags": ["合同"]} }
```

#### 3.2.9 问答
`POST /kb/{kb_id}/ask`

Response:
```
{ "answer": "...", "citations": [ {"file": "合同.pdf", "page": 12, "snippet": "...", "source_deleted": false} ] }
```

#### 3.2.10 统计
`GET /kb/{kb_id}/stats`

#### 3.2.11 错误列表
`GET /kb/{kb_id}/errors?type=ocr`

#### 3.2.12 路径映射管理
`GET /config/paths`
`POST /config/paths`
`DELETE /config/paths`

## 4. CLI 与 API 对应关系
- `ragkb kb create` → `POST /kb`
- `ragkb ingest add` → `POST /kb/{kb_id}/ingest`
- `ragkb ingest scan` → `POST /kb/{kb_id}/scan`
- `ragkb index build` → `POST /kb/{kb_id}/index/build`
- `ragkb ask` → `POST /kb/{kb_id}/ask`
- `ragkb stats` → `GET /kb/{kb_id}/stats`

## 5. 输出格式规范
### 5.1 text 输出
- 面向人工阅读，包含可复制的引用信息与文件路径。

### 5.2 json 输出
- 面向脚本与自动化集成，字段固定。

示例：
```
{ "status": "ok", "data": { "answer": "...", "citations": [] } }
```

## 6. 错误码规范（建议）
- `40001` 参数错误
- `40401` KB 不存在
- `40402` 文档不存在
- `40901` 资源冲突（重复 KB）
- `50001` OCR 失败
- `50002` 索引失败

## 7. 与当前项目的对齐建议
- `process_id` 可视作 `kb_id` 的别名（保持兼容）。
- 数据目录延用：
  - `data/processed/<kb_id>/...`
  - `index/<kb_id>/...`
  - `data/log/<kb_id>/...`
  - `data/manifest/<kb_id>/...`（含 `kb_paths.json` 及源文件状态清单）
 - 原始文件保留在用户指定的系统路径，不复制进项目目录。

## 8. 待确认点
- CLI 名称是否固定为 `ragkb`？
- 是否需要内置本地 HTTP 服务（FastAPI）？
- 默认输出语言（中文/英文）？
