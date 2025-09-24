# MCP 智能邮件处理系统设计文档

## 1. 系统概述
本系统基于 MCP（Model Context Protocol）构建，旨在实现对邮件的智能处理和自动化任务执行。通过模块化设计和上下文感知能力，系统能够解析邮件意图、调用适配工具完成任务，并将结果反馈给用户。

## 2. 系统架构
系统由 6 个核心模块组成：
1. **邮件输入监听模块**：监听邮箱，拉取新邮件并转化为 ContextObject。
2. **上下文管理引擎**：存储、检索、压缩邮件上下文，生成 PromptBlock。
3. **任务编排与路由模块**：解析意图，生成任务并路由到工具或推理层。
4. **语义推理接口层**：与 LLM/MCP 客户端通信，执行意图识别、摘要、分类、草稿生成。
5. **邮件处理工具模块**：执行具体动作，如回复草稿生成、日程提取、附件分类。
6. **任务输出与反馈模块**：收集执行结果，通知用户并记录日志。

## 3. 数据流
1. **监听阶段**：新邮件到达 → 转换为 ContextObject。
2. **上下文构建**：上下文管理引擎检索相关邮件，生成 PromptBlock。
3. **任务解析与编排**：任务编排模块生成任务，调用 CreateMessageRequestParams。
4. **推理调用**：语义推理接口层调用 LLM/MCP，返回 ToolResult。
5. **工具执行**：邮件处理工具执行具体动作，返回执行结果。
6. **反馈输出**：结果写回上下文并通过邮件/消息通知用户。

## 4. 关键实现要点
- **上下文对象设计**：使用 ContextObject 封装邮件字段，保持结构化输入。
- **PromptBlock 构建**：支持历史邮件拼接、上下文压缩，降低 token 成本。
- **任务抽象**：每个任务独立定义，便于扩展和优先级调度。
- **Pydantic 工具参数**：保证输入参数类型安全与自动验证。
- **异步调用与回调**：支持 call_tool() 异步执行，采样回调 sampling_callback，RequestContext 用于日志追踪。

## 5. 技术选型
- **协议与监听**：IMAP + OAuth2，支持 Gmail API webhook。
- **向量数据库**：FAISS/Milvus 用于上下文检索。
- **任务队列**：Celery/RabbitMQ 管理任务执行。
- **推理接口**：MCP ClientSession + LLM（本地 + 云端混合）。
- **日志与监控**：统一 Logging 模块，支持错误重试与告警。

## 6. API 设计说明

### 6.1 ContextObject
```json
{
  "id": "string",
  "sender": "user@example.com",
  "subject": "string",
  "body": "string",
  "attachments": ["file1.pdf", "file2.docx"],
  "timestamp": "2025-09-24T08:30:00Z"
}
```

### 6.2 PromptBlock
```json
{
  "context": ["previous email content..."],
  "current": "current email content...",
  "metadata": {
    "thread_id": "string",
    "priority": "high|medium|low"
  }
}
```

### 6.3 CreateMessageRequestParams
```json
{
  "task_type": "summarize|classify|generate_reply",
  "input": "string or structured payload",
  "context": "PromptBlock",
  "tools": ["tool_name_1", "tool_name_2"]
}
```

### 6.4 ToolResult
```json
{
  "tool": "tool_name",
  "status": "success|failed",
  "output": "structured result or message",
  "logs": ["execution detail..."],
  "timestamp": "2025-09-24T08:31:00Z"
}
```

### 6.5 调用示例
```python
from mcp.client import ClientSession
from tools import generate_reply

session = ClientSession()
request = CreateMessageRequestParams(
    task_type="generate_reply",
    input="Please generate a polite reply",
    context=prompt_block,
    tools=["generate_reply"]
)

result = session.call_tool(generate_reply, request)
print(result.output)
```

## 7. 模块优势
- 提升语义独立性：上下文与推理解耦，可替换模型。
- 增强可维护性：模块化设计，升级影响最小。
- 优化上下文链：向量检索和压缩减少 token 消耗。
- 支持异步工具调用：提高吞吐量。
- 流程可扩展：轻松新增工具或外部集成。

## 8. 后续计划
- 增加多用户上下文隔离与权限控制。
- 接入外部知识库增强邮件理解。
- 引入强化学习优化任务优先级排序。
- 提供可视化控制台展示任务流与日志。

## 9. 错误处理与日志设计（新增）

### 9.1 异常分类与处理策略
- **输入层错误（ValidationError）**：由 Pydantic/参数校验触发；返回 4xx 给上游，记录详细字段错误。
- **业务处理错误（BusinessError）**：任务逻辑或工具执行中出现的业务异常；尝试有限次重试（见重试策略），失败后写入死信队列（DLQ）并通知人工介入。
- **外部依赖错误（DependencyError）**：如向量库、邮件服务或第三方 API 不可用；采用指数退避（exponential backoff）重试，配合熔断器（circuit breaker）保护系统。
- **系统错误（SystemError）**：内存/磁盘/数据库等资源异常；触发紧急告警，并切换降级策略（如使用缓存的上下文或只做入队日志）。

### 9.2 重试与降级机制
- **指数退避 + 抖动**：初始间隔 1s，最大重试 5 次；在重试间加入随机抖动避免同质流量。
- **幂等设计**：所有 Task 都必须有唯一 `task_id`，工具执行应检查 `task_id` 防止重复执行。
- **死信队列（DLQ）**：超过重试次数的消息写入 DLQ，供离线人工或补偿服务处理。
- **熔断器**：对关键外部依赖设置熔断阈值（如 50% 错误率或延迟超过阈值），在熔断期间快速失败并记录降级原因。

### 9.3 日志与追踪规范
- **统一日志格式（JSON）**，包含字段：`timestamp, level, component, task_id, request_id, user_id, thread_id, message, error, duration_ms`。
- **关联 ID**：每个请求和任务带 `request_id` 与 `task_id`，跨模块透传，便于链路追踪（traceability）。
- **采样策略**：对 TRACE 级日志采用 1% 采样，对 ERROR/WARN 全量采集；采样回调（`sampling_callback`）用于在采样到重要事件时导出完整上下文。
- **日志存储**：短期高频日志进入 ELK/EFK 集群，长时归档到对象存储（如 S3）并做索引。

### 9.4 监控与告警
- **关键指标**：任务成功率、平均处理延时（P50/P95/P99）、队列积压长度、外部依赖错误率、DLQ 增长率。
- **SLO / SLA**：例如 99% 的生成回复在 5s 内返回；超出阈值触发 PagerDuty 告警。
- **仪表盘**：Grafana 面板展示关键指标、调用链延时分布、最近错误样本。
- **告警策略**：对系统级别错误设置实时告警；对退化级别问题设邮件/日终报告。

### 9.5 采样回调（sampling_callback）与 RequestContext
- **RequestContext**：在调用 LLM / Tool 时携带 `request_id, user_id, task_id, metadata`，并在日志与指标中透传。
- **sampling_callback**：实现钩子，在满足条件（如出错、延迟高、罕见意图）时抓取完整上下文并上报，便于线下复现与模型调优。
- **示例用途**：模型偏差检测、少样本错误回溯、A/B 测试样本抽取。

### 9.6 可观测性与可测试性
- **链路追踪（Distributed Tracing）**：集成 OpenTelemetry，捕获跨模块调用的 Span 与 Trace。
- **契约测试（Contract Tests）**：对工具接口使用契约测试保证输入输出格式稳定。
- **混沌测试（Chaos Engineering）**：在非生产环境验证熔断、重试、DLQ 行为。
- **回归日志分析**：提供便捷查询入口，从采样日志中生成复现脚本与单元测试用例。

### 9.7 安全与合规
- **敏感信息脱敏**：在日志和采样数据中对 PII（邮箱、姓名、身份证号等）进行脱敏或哈希处理。
- **访问控制**：日志访问与采样数据导出需要审计和权限控制。
- **合规保留策略**：根据法规设定日志保留期并支持删除请求（如 GDPR 数据删除）。

---

（文档已追加“错误处理与日志设计”章节，包含异常分类、重试策略、日志规范、监控与采样回调实践等内容。）

