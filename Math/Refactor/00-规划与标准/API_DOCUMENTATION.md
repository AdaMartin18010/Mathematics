# 数学格式修复项目 API 文档

## 概述
数学格式修复项目提供完整的API接口，支持数学格式的检查、修复和批量处理。

## 基础信息
- **API版本**: v1.0.0
- **基础URL**: http://localhost:8000
- **认证方式**: 无 (开发版本)

## 端点列表

### 1. 格式检查
**POST** `/api/check`

检查数学公式格式问题。

**请求参数**:
```json
{
    "content": "数学公式内容",
    "format": "markdown"
}
```

**响应**:
```json
{
    "status": "success",
    "issues": [
        {
            "type": "format_error",
            "message": "格式错误描述",
            "position": 10,
            "suggestion": "修复建议"
        }
    ]
}
```

### 2. 格式修复
**POST** `/api/fix`

修复数学公式格式问题。

**请求参数**:
```json
{
    "content": "原始内容",
    "format": "markdown",
    "rules": ["rule1", "rule2"]
}
```

**响应**:
```json
{
    "status": "success",
    "fixed_content": "修复后的内容",
    "changes": [
        {
            "type": "fix",
            "description": "修复描述",
            "position": 10
        }
    ]
}
```

### 3. 批量处理
**POST** `/api/batch`

批量处理多个文件。

**请求参数**:
```json
{
    "files": [
        {
            "name": "file1.md",
            "content": "文件内容"
        }
    ],
    "options": {
        "format": "markdown",
        "parallel": true
    }
}
```

**响应**:
```json
{
    "status": "success",
    "results": [
        {
            "file": "file1.md",
            "status": "success",
            "fixed_content": "修复后的内容"
        }
    ]
}
```

### 4. 状态查询
**GET** `/api/status`

获取服务状态。

**响应**:
```json
{
    "status": "running",
    "version": "1.0.0",
    "uptime": 3600,
    "requests_processed": 100
}
```

### 5. 统计信息
**GET** `/api/stats`

获取处理统计信息。

**响应**:
```json
{
    "total_files": 1000,
    "success_rate": 98.5,
    "average_time": 0.8,
    "error_rate": 1.5
}
```

## 错误处理

### 错误响应格式
```json
{
    "status": "error",
    "error_code": "INVALID_FORMAT",
    "message": "错误描述",
    "details": {}
}
```

### 常见错误码
- `INVALID_FORMAT`: 格式无效
- `PROCESSING_ERROR`: 处理错误
- `FILE_TOO_LARGE`: 文件过大
- `UNSUPPORTED_FORMAT`: 不支持的格式

## 使用示例

### Python示例
```python
import requests

# 检查格式
response = requests.post('http://localhost:8000/api/check', json={
    'content': 'x^2 + y^2 = z^2',
    'format': 'markdown'
})

# 修复格式
response = requests.post('http://localhost:8000/api/fix', json={
    'content': 'x^2 + y^2 = z^2',
    'format': 'markdown'
})
```

### JavaScript示例
```javascript
// 检查格式
fetch('http://localhost:8000/api/check', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        content: 'x^2 + y^2 = z^2',
        format: 'markdown'
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 限制和注意事项

1. **文件大小限制**: 单个文件不超过10MB
2. **并发限制**: 最多支持100个并发请求
3. **格式支持**: 目前支持Markdown、LaTeX、HTML格式
4. **处理时间**: 单个文件处理时间不超过30秒

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基础格式检查和修复
- 提供RESTful API接口
- 支持批量处理功能
