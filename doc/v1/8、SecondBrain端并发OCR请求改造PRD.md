# 8、SecondBrain端并发OCR请求改造PRD

## 1. 背景与问题
- 当前 `ingest` 流程在 OCR 阶段以“单文件、单页串行”方式调用远程 OCR 服务。
- 在 PDF 页数较多时，网络往返、服务排队和单请求开销会被放大，导致整体入库吞吐低。
- 现状即使 OCR 服务端 GPU 已高负载，客户端仍可能因串行调度导致总耗时偏高。

## 2. 目标
- 将 SecondBrain 端 OCR 请求改造为“按页并发，可配并发度”。
- 在不改变现有入库结果语义（文本内容、页序、错误语义）的前提下提升吞吐。
- 支持通过环境变量快速调参，适配不同网络与 OCR 服务能力。

## 3. 非目标
- 不改 OCR 服务端识别模型与推理代码。
- 不改 chunk、embedding、索引写入等后续业务逻辑。
- 不引入分布式任务队列（仅进程内并发）。

## 4. 现状流程（简化）
1. `ingest_kb` 遍历文件。
2. PDF 进入 `DocumentTextExtractor._extract_pdf_pages`。
3. 每页转图片后，逐页调用 `ocr_image_bytes`（串行）。
4. 所有页 OCR 完成后再拼接文本并继续入库。

问题点：第 3 步串行，无法利用远端 OCR 服务可并发处理的能力。

## 5. 目标方案总览
核心策略：`页级任务池 + 有界并发 + 顺序回填 + 失败隔离`。

- 页级任务池：每页图片作为一个 OCR 任务。
- 有界并发：使用 `ThreadPoolExecutor(max_workers=N)`，`N` 由配置控制。
- 顺序回填：任务可乱序完成，但最终按页码顺序拼接文本，保证语义稳定。
- 失败隔离：单页失败仅影响该页，记录错误并继续处理其余页面。

## 6. 并发逻辑设计
### 6.1 并发粒度
- 粒度：PDF 页级。
- 原因：页级任务天然独立，最适合远程 HTTP OCR 并发。

### 6.2 执行模型
1. 将 PDF 转为 `(page_no, image_bytes, filename)` 列表。
2. 预分配 `page_texts = [""] * page_count`。
3. 提交所有页任务到线程池（受 `max_workers` 限制）。
4. `as_completed` 收集结果：
   - 成功：写入 `page_texts[page_no]`。
   - 失败：记录日志与计数，`page_texts[page_no]` 保持空字符串。
5. 全部任务完成后按 `page_no` 顺序合并文本。

### 6.3 顺序保证
- 虽然请求并发发出且返回乱序，但通过页码索引写回，最终输出顺序与原文页序一致。

### 6.4 错误处理
- 单页异常（超时、HTTP 5xx、连接异常）不终止整份 PDF。
- 对失败页记录：
  - 文件路径
  - 页码
  - 错误摘要
- 若失败页占比超过阈值（可选，如 80%），将该文件标记为 `error`，避免写入低质量内容。

## 7. 配置项（新增）
- `OCR_CONCURRENCY`：页级并发度，默认 `4`。
- `OCR_PAGE_TIMEOUT`：单页 OCR 超时时间（秒），默认沿用 `OCR_TIMEOUT`。
- `OCR_MAX_RETRIES`：单页重试次数，默认 `1`。
- `OCR_RETRY_BACKOFF_MS`：重试退避毫秒，默认 `300`。

说明：
- 低网络延迟或服务端能力强时可提高 `OCR_CONCURRENCY`（如 `6~12`）。
- 服务端已接近饱和时，过高并发会增加排队和超时，需压回。

## 8. 关键实现点
### 8.1 涉及模块
- `src/util/document_extractor.py`
  - 改造 `_extract_pdf_pages` 为并发版本（或新增 `_extract_pdf_pages_concurrent`）。
- `src/util/remote_ocr.py`
  - 保持线程安全（当前无共享可变状态，主要是 HTTP 请求调用）。
  - 增加更清晰的超时与重试日志。
- `src/app/settings.py`
  - 增加并解析上述配置项。

### 8.2 伪代码
```python
page_items = list(tool.iter_pdf_images(pdf_path))
page_texts = [""] * len(page_items)

with ThreadPoolExecutor(max_workers=settings.ocr_concurrency) as ex:
    fut_map = {}
    for page_no, img, name in page_items:
        fut = ex.submit(run_ocr_with_retry, img, name, timeout=settings.ocr_page_timeout)
        fut_map[fut] = page_no

    for fut in as_completed(fut_map):
        page_no = fut_map[fut]
        try:
            lines = fut.result()
            page_texts[page_no] = "\n".join(lines)
        except Exception as e:
            log_warn(pdf_path, page_no, e)
            page_texts[page_no] = ""
```

## 9. 效率提升原理
- 串行模式总耗时近似：`T_total ~= sum(T_page_i)`。
- 并发模式总耗时近似：`T_total ~= max(分组内最慢页耗时) * 组数 + 调度开销`。
- 当 OCR 为远程 HTTP 调用时，并发可显著摊薄等待时间与单请求固定开销。
- 对多页文档，通常可获得 `1.8x~4x` 吞吐提升（取决于网络、服务端并发能力、页复杂度分布）。

## 10. 兼容性与风险
- 风险1：并发过高导致 OCR 服务端排队/超时升高。
  - 规避：默认低并发，逐步压测调优。
- 风险2：短时请求峰值可能触发服务端限流。
  - 规避：可选加入客户端限速与指数退避。
- 风险3：错误页变多导致文本缺失。
  - 规避：失败率阈值与失败页统计告警。

## 11. 验收标准
- 功能一致性：
  - 同一 PDF 在串行与并发模式下，页序一致。
  - 正常页 OCR 文本不因并发而错位或串页。
- 稳定性：
  - 单页失败不导致整文件失败（除超过失败阈值）。
- 性能：
  - 在 50 页以上 PDF 样本上，总耗时较现网串行模式下降至少 `30%`。
  - OCR 服务端错误率不高于基线 `+5%`。

## 12. 发布与回滚
- 发布步骤：
1. 增加配置项，默认 `OCR_CONCURRENCY=4`。
2. 灰度开启：先在单 KB 测试，再全量。
3. 观察 24 小时：耗时、超时率、失败页比例。
- 回滚策略：
  - 将 `OCR_CONCURRENCY=1` 即回到串行行为，无需代码回退。

## 13. 里程碑
1. M1（0.5 天）：设置项与并发框架接入。
2. M2（0.5 天）：重试、日志、失败统计完善。
3. M3（0.5 天）：压测与参数建议固化。

