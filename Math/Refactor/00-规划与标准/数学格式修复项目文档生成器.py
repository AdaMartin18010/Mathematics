#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复项目文档生成器
自动生成项目文档、API文档和使用手册
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

class DocumentationGenerator:
    """文档生成器"""
    
    def __init__(self):
        self.project_info = {
            'name': '数学格式修复项目',
            'version': '1.0.0',
            'description': '专业的数学格式修复工具，支持多种格式规范和批量处理',
            'author': '数学格式修复团队',
            'license': 'MIT',
            'repository': 'https://github.com/math-format-fix/project'
        }
        
        self.modules = {
            '核心模块': {
                'file': '数学格式修复核心模块.py',
                'description': '核心修复功能和格式检查',
                'functions': [
                    'fix_text(text: str) -> str',
                    'check_format(text: str) -> List[Dict]',
                    'update_config(config: Dict) -> None'
                ]
            },
            '配置管理器': {
                'file': '数学格式修复配置管理器.py',
                'description': '配置文件管理和验证',
                'functions': [
                    'load_config(config_path: str) -> None',
                    'get_config() -> Dict',
                    'validate_config(config: Dict) -> bool',
                    'export_config(output_path: str) -> None'
                ]
            },
            '批量处理工具': {
                'file': '数学格式修复批量处理工具.py',
                'description': '大规模文件处理和进度监控',
                'functions': [
                    'process_files(file_list: List[str]) -> Dict',
                    'monitor_progress() -> Dict',
                    'generate_report() -> str'
                ]
            },
            '命令行工具': {
                'file': '数学格式修复命令行工具.py',
                'description': '命令行界面和操作',
                'functions': [
                    'process_single_file(file_path: str) -> bool',
                    'process_directory(dir_path: str) -> Dict',
                    'check_format(file_path: str) -> Dict'
                ]
            },
            'Web界面': {
                'file': '数学格式修复Web界面.py',
                'description': '基于Flask的Web用户界面',
                'functions': [
                    'upload_file() -> Response',
                    'process_batch() -> Response',
                    'download_result() -> Response'
                ]
            },
            'API服务': {
                'file': '数学格式修复API服务.py',
                'description': 'RESTful API接口服务',
                'functions': [
                    'POST /api/check - 检查文本格式',
                    'POST /api/fix - 修复文本格式',
                    'POST /api/tasks - 创建批量任务',
                    'GET /api/tasks/{task_id} - 获取任务状态'
                ]
            }
        }
    
    def generate_readme(self, output_path: str = 'README.md'):
        """生成项目README文档"""
        content = f"""# {self.project_info['name']}

{self.project_info['description']}

## 项目信息

- **版本**: {self.project_info['version']}
- **作者**: {self.project_info['author']}
- **许可证**: {self.project_info['license']}
- **仓库**: {self.project_info['repository']}

## 功能特性

### 核心功能
- ✅ 数学公式格式修复
- ✅ 多种格式规范支持
- ✅ 批量文件处理
- ✅ 实时进度监控
- ✅ 详细处理报告

### 用户界面
- 🖥️ 命令行工具
- 🌐 Web界面
- 📱 RESTful API
- ⚙️ 配置管理

### 质量保证
- 🧪 全面测试套件
- 📊 性能监控
- 🔍 格式检查
- 📈 统计报告

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用
```bash
# 修复单个文件
python 数学格式修复命令行工具.py fix-file input.md -o output.md

# 批量处理目录
python 数学格式修复命令行工具.py fix-dir ./docs -r

# 检查格式问题
python 数学格式修复命令行工具.py check input.md
```

### Web界面
```bash
python 数学格式修复Web界面.py
# 访问 http://localhost:5000
```

### API服务
```bash
python 数学格式修复API服务.py
# API地址: http://localhost:8000
```

## 项目结构

```
Math/Refactor/00-规划与标准/
├── 数学格式修复核心模块.py          # 核心修复功能
├── 数学格式修复配置管理器.py          # 配置管理
├── 数学格式修复批量处理工具.py        # 批量处理
├── 数学格式修复命令行工具.py          # 命令行界面
├── 数学格式修复Web界面.py            # Web用户界面
├── 数学格式修复API服务.py            # API服务
├── 数学格式修复测试套件.py            # 测试套件
├── 数学格式修复项目文档生成器.py      # 文档生成
├── 数学格式修复项目完整规范总结文档.md  # 项目规范
├── 数学格式修复项目最终总结文档.md      # 项目总结
└── 数学格式修复项目完整索引文档.md      # 项目索引
```

## 配置说明

### 基本配置
```json
{{
    "rules": {{
        "dollar_signs": true,
        "brackets": true,
        "spacing": true
    }},
    "settings": {{
        "output_format": "markdown",
        "preserve_original": true
    }}
}}
```

### 高级配置
- 自定义修复规则
- 输出格式设置
- 性能优化参数
- 错误处理策略

## 测试

运行完整测试套件：
```bash
python 数学格式修复测试套件.py
```

测试覆盖：
- ✅ 单元测试
- ✅ 集成测试
- ✅ 性能测试
- ✅ 错误处理测试
- ✅ 回归测试

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 {self.project_info['license']} 许可证。

## 联系方式

- 项目主页: {self.project_info['repository']}
- 问题反馈: {self.project_info['repository']}/issues
- 讨论区: {self.project_info['repository']}/discussions

---

*最后更新: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"README文档已生成: {output_path}")
    
    def generate_api_docs(self, output_path: str = 'API文档.md'):
        """生成API文档"""
        content = f"""# 数学格式修复API文档

## 概述

数学格式修复API提供RESTful接口，支持文本格式检查和修复功能。

**基础URL**: `http://localhost:8000`

## 认证

目前API不需要认证，所有接口都是公开的。

## 响应格式

所有API响应都使用JSON格式，包含以下字段：

```json
{{
    "success": true,
    "data": {{}},
    "message": "操作成功",
    "timestamp": "2024-01-01T12:00:00Z"
}}
```

## 接口列表

### 1. 检查文本格式

**POST** `/api/check`

检查文本中的数学格式问题。

**请求参数**:
```json
{{
    "text": "要检查的文本内容",
    "config": {{
        "rules": {{
            "dollar_signs": true,
            "brackets": true
        }}
    }}
}}
```

**响应示例**:
```json
{{
    "success": true,
    "data": {{
        "issues": [
            {{
                "type": "single_dollar",
                "message": "发现单美元符号",
                "line": 1,
                "column": 10
            }}
        ],
        "total_issues": 1
    }},
    "message": "检查完成"
}}
```

### 2. 修复文本格式

**POST** `/api/fix`

修复文本中的数学格式问题。

**请求参数**:
```json
{{
    "text": "要修复的文本内容",
    "config": {{
        "rules": {{
            "dollar_signs": true,
            "brackets": true
        }}
    }}
}}
```

**响应示例**:
```json
{{
    "success": true,
    "data": {{
        "original": "原始文本",
        "fixed": "修复后的文本",
        "changes": [
            {{
                "type": "dollar_signs",
                "count": 2
            }}
        ]
    }},
    "message": "修复完成"
}}
```

### 3. 创建批量任务

**POST** `/api/tasks`

创建批量处理任务。

**请求参数**:
```json
{{
    "files": [
        "file1.md",
        "file2.md"
    ],
    "config": {{
        "rules": {{
            "dollar_signs": true
        }}
    }}
}}
```

**响应示例**:
```json
{{
    "success": true,
    "data": {{
        "task_id": "task_123456",
        "status": "created",
        "total_files": 2
    }},
    "message": "任务创建成功"
}}
```

### 4. 获取任务状态

**GET** `/api/tasks/{task_id}`

获取批量任务的处理状态。

**响应示例**:
```json
{{
    "success": true,
    "data": {{
        "task_id": "task_123456",
        "status": "processing",
        "progress": {{
            "completed": 1,
            "total": 2,
            "percentage": 50
        }},
        "results": [
            {{
                "file": "file1.md",
                "status": "completed",
                "issues_fixed": 3
            }}
        ]
    }},
    "message": "任务状态获取成功"
}}
```

### 5. 获取任务结果

**GET** `/api/tasks/{task_id}/result`

获取批量任务的最终结果。

**响应示例**:
```json
{{
    "success": true,
    "data": {{
        "task_id": "task_123456",
        "status": "completed",
        "summary": {{
            "total_files": 2,
            "successful": 2,
            "failed": 0,
            "total_issues": 5
        }},
        "results": [
            {{
                "file": "file1.md",
                "status": "completed",
                "issues_fixed": 3,
                "output_file": "file1_fixed.md"
            }}
        ]
    }},
    "message": "任务结果获取成功"
}}
```

### 6. 取消任务

**DELETE** `/api/tasks/{task_id}`

取消正在进行的批量任务。

**响应示例**:
```json
{{
    "success": true,
    "data": {{
        "task_id": "task_123456",
        "status": "cancelled"
    }},
    "message": "任务已取消"
}}
```

### 7. 获取统计信息

**GET** `/api/stats`

获取系统统计信息。

**响应示例**:
```json
{{
    "success": true,
    "data": {{
        "total_requests": 1000,
        "successful_requests": 950,
        "failed_requests": 50,
        "average_response_time": 0.5,
        "active_tasks": 2,
        "completed_tasks": 150
    }},
    "message": "统计信息获取成功"
}}
```

## 错误码

| 错误码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

## 使用示例

### Python示例

```python
import requests

# 检查文本格式
response = requests.post('http://localhost:8000/api/check', json={{
    'text': '这是一个公式: $x^2$',
    'config': {{'rules': {{'dollar_signs': True}}}}
}})
print(response.json())

# 修复文本格式
response = requests.post('http://localhost:8000/api/fix', json={{
    'text': '这是一个公式: $x^2$',
    'config': {{'rules': {{'dollar_signs': True}}}}
}})
print(response.json())
```

### JavaScript示例

```javascript
// 检查文本格式
fetch('http://localhost:8000/api/check', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
        text: '这是一个公式: $x^2$',
        config: {{rules: {{dollar_signs: true}}}}
    }})
}})
.then(response => response.json())
.then(data => console.log(data));
```

## 限制说明

- 单次请求文本大小限制: 10MB
- 批量任务文件数量限制: 1000个
- 任务保留时间: 24小时
- 并发请求限制: 100个/分钟

---

*文档生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"API文档已生成: {output_path}")
    
    def generate_user_manual(self, output_path: str = '用户手册.md'):
        """生成用户手册"""
        content = f"""# 数学格式修复工具用户手册

## 目录

1. [简介](#简介)
2. [安装配置](#安装配置)
3. [基本使用](#基本使用)
4. [高级功能](#高级功能)
5. [配置说明](#配置说明)
6. [故障排除](#故障排除)
7. [常见问题](#常见问题)

## 简介

数学格式修复工具是一个专业的数学文档格式处理工具，能够自动检测和修复数学公式的格式问题，支持多种输入输出格式。

### 主要功能

- 🔧 **自动修复**: 智能识别和修复数学公式格式问题
- 📁 **批量处理**: 支持大量文件的批量处理
- 🎯 **多种格式**: 支持Markdown、LaTeX等多种格式
- 📊 **详细报告**: 提供详细的处理报告和统计信息
- ⚙️ **灵活配置**: 支持自定义修复规则和参数

### 适用场景

- 学术论文格式标准化
- 技术文档格式修复
- 数学教材格式整理
- 科研报告格式统一

## 安装配置

### 系统要求

- Python 3.7+
- 内存: 2GB+
- 磁盘空间: 100MB+

### 安装步骤

1. **下载项目**
```bash
git clone {self.project_info['repository']}
cd math-format-fix
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **验证安装**
```bash
python 数学格式修复命令行工具.py --help
```

### 配置文件

创建配置文件 `config.json`:
```json
{{
    "rules": {{
        "dollar_signs": true,
        "brackets": true,
        "spacing": true,
        "alignment": true
    }},
    "settings": {{
        "output_format": "markdown",
        "preserve_original": true,
        "backup_files": true
    }},
    "performance": {{
        "max_file_size": "10MB",
        "batch_size": 100,
        "timeout": 300
    }}
}}
```

## 基本使用

### 命令行工具

#### 修复单个文件
```bash
python 数学格式修复命令行工具.py fix-file input.md -o output.md
```

#### 批量处理目录
```bash
python 数学格式修复命令行工具.py fix-dir ./docs -r -o ./fixed_docs
```

#### 检查格式问题
```bash
python 数学格式修复命令行工具.py check input.md
```

#### 验证配置文件
```bash
python 数学格式修复命令行工具.py config config.json
```

### Web界面

1. **启动服务**
```bash
python 数学格式修复Web界面.py
```

2. **访问界面**
打开浏览器访问 `http://localhost:5000`

3. **使用功能**
- 上传单个文件进行修复
- 批量上传多个文件
- 实时查看处理进度
- 下载处理结果

### API服务

1. **启动API服务**
```bash
python 数学格式修复API服务.py
```

2. **API调用示例**
```python
import requests

# 检查文本格式
response = requests.post('http://localhost:8000/api/check', json={{
    'text': '这是一个公式: $x^2$'
}})

# 修复文本格式
response = requests.post('http://localhost:8000/api/fix', json={{
    'text': '这是一个公式: $x^2$'
}})
```

## 高级功能

### 自定义修复规则

在配置文件中定义自定义规则：

```json
{{
    "custom_rules": [
        {{
            "name": "custom_rule_1",
            "pattern": "\\\\\\[([^\\]]+)\\\\\\]",
            "replacement": "$$$1$$",
            "description": "将LaTeX行间公式转换为Markdown格式"
        }}
    ]
}}
```

### 批量处理优化

对于大量文件，可以使用以下优化策略：

1. **并行处理**
```bash
python 数学格式修复批量处理工具.py --parallel --workers 4
```

2. **内存优化**
```bash
python 数学格式修复批量处理工具.py --memory-limit 2GB
```

3. **进度监控**
```bash
python 数学格式修复批量处理工具.py --monitor --log-level DEBUG
```

### 报告生成

生成详细的处理报告：

```bash
python 数学格式修复命令行工具.py fix-dir ./docs --report report.json
```

报告内容包括：
- 处理文件统计
- 修复问题详情
- 性能指标
- 错误信息

## 配置说明

### 修复规则配置

| 规则 | 说明 | 默认值 |
|------|------|--------|
| dollar_signs | 美元符号处理 | true |
| brackets | 括号处理 | true |
| spacing | 间距处理 | true |
| alignment | 对齐处理 | true |
| comments | 注释处理 | false |

### 输出设置

| 设置 | 说明 | 选项 |
|------|------|------|
| output_format | 输出格式 | markdown, latex, html |
| preserve_original | 保留原文件 | true, false |
| backup_files | 备份文件 | true, false |
| encoding | 文件编码 | utf-8, gbk, gb2312 |

### 性能设置

| 设置 | 说明 | 默认值 |
|------|------|--------|
| max_file_size | 最大文件大小 | 10MB |
| batch_size | 批处理大小 | 100 |
| timeout | 超时时间 | 300秒 |
| memory_limit | 内存限制 | 2GB |

## 故障排除

### 常见错误

1. **文件编码错误**
```
错误: UnicodeDecodeError
解决: 检查文件编码，使用UTF-8编码
```

2. **内存不足**
```
错误: MemoryError
解决: 减少批处理大小，增加内存限制
```

3. **超时错误**
```
错误: TimeoutError
解决: 增加超时时间，检查文件大小
```

### 日志分析

查看详细日志：
```bash
tail -f math_format_fix.log
```

日志级别设置：
```bash
python 数学格式修复命令行工具.py --verbose
```

### 性能优化

1. **文件大小优化**
- 分割大文件
- 压缩文件
- 清理无用内容

2. **处理速度优化**
- 使用SSD存储
- 增加内存
- 并行处理

3. **网络优化**
- 使用本地服务器
- 优化网络带宽
- 缓存处理结果

## 常见问题

### Q: 工具支持哪些文件格式？
A: 目前支持Markdown(.md)、LaTeX(.tex)、HTML(.html)等格式。

### Q: 如何处理超大文件？
A: 建议将大文件分割成小块，或使用批处理模式。

### Q: 修复后的文件在哪里？
A: 默认在当前目录生成，可通过-o参数指定输出路径。

### Q: 如何自定义修复规则？
A: 在配置文件的custom_rules部分添加自定义规则。

### Q: 支持哪些数学公式格式？
A: 支持LaTeX、MathML、AsciiMath等多种格式。

### Q: 如何处理中文文档？
A: 确保文件使用UTF-8编码，工具会自动处理中文内容。

### Q: 批量处理时如何监控进度？
A: 使用--monitor参数或查看Web界面的实时进度。

### Q: 如何备份原始文件？
A: 在配置文件中设置backup_files为true。

---

*手册生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"用户手册已生成: {output_path}")
    
    def generate_development_guide(self, output_path: str = '开发指南.md'):
        """生成开发指南"""
        content = f"""# 数学格式修复项目开发指南

## 项目概述

数学格式修复项目是一个开源的数学文档格式处理工具，采用模块化设计，支持多种用户界面和部署方式。

## 技术架构

### 核心架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户界面层     │    │    业务逻辑层    │    │    数据访问层    │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ - 命令行工具     │    │ - 格式修复引擎   │    │ - 文件系统      │
│ - Web界面       │    │ - 配置管理器     │    │ - 数据库        │
│ - API服务       │    │ - 批量处理器     │    │ - 缓存系统      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 模块说明

"""

        # 添加模块说明
        for module_name, module_info in self.modules.items():
            content += f"""
#### {module_name}

**文件**: `{module_info['file']}`

**描述**: {module_info['description']}

**主要功能**:
"""
            for func in module_info['functions']:
                content += f"- `{func}`\n"
            content += "\n"

        content += f"""
## 开发环境

### 环境要求

- Python 3.7+
- Git
- IDE (推荐PyCharm或VS Code)
- 测试框架: unittest, pytest

### 环境搭建

1. **克隆项目**
```bash
git clone {self.project_info['repository']}
cd math-format-fix
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **运行测试**
```bash
python 数学格式修复测试套件.py
```

## 代码规范

### Python代码规范

遵循PEP 8规范：

```python
# 导入顺序
import os
import sys
from typing import List, Dict

# 类定义
class MathFormatFixer:
    """数学格式修复器"""
    
    def __init__(self):
        self.config = {{}}
    
    def fix_text(self, text: str) -> str:
        """修复文本格式"""
        # 实现代码
        pass
```

### 文档规范

- 所有函数必须有docstring
- 使用中文注释
- 重要功能需要详细说明

### 测试规范

- 单元测试覆盖率 > 90%
- 每个功能必须有对应测试
- 测试用例要覆盖边界情况

## 开发流程

### 功能开发

1. **创建分支**
```bash
git checkout -b feature/new-feature
```

2. **开发功能**
- 编写代码
- 添加测试
- 更新文档

3. **提交代码**
```bash
git add .
git commit -m "feat: 添加新功能"
git push origin feature/new-feature
```

4. **创建PR**
- 提交Pull Request
- 代码审查
- 合并到主分支

### 问题修复

1. **创建Issue**
- 描述问题
- 提供复现步骤
- 添加错误日志

2. **修复问题**
- 定位问题
- 编写修复代码
- 添加测试用例

3. **验证修复**
- 运行测试
- 手动验证
- 更新文档

## 测试指南

### 单元测试

```python
import unittest

class TestMathFormatFixer(unittest.TestCase):
    def setUp(self):
        self.fixer = MathFormatFixer()
    
    def test_fix_text_basic(self):
        """测试基本文本修复"""
        result = self.fixer.fix_text("$x^2$")
        self.assertEqual(result, "$$x^2$$")
```

### 集成测试

```python
def test_end_to_end_workflow(self):
    """测试端到端工作流程"""
    # 1. 加载配置
    # 2. 处理文件
    # 3. 验证结果
    pass
```

### 性能测试

```python
def test_large_file_performance(self):
    """测试大文件性能"""
    large_text = "测试文本 " * 10000
    start_time = time.time()
    result = self.fixer.fix_text(large_text)
    end_time = time.time()
    
    self.assertLess(end_time - start_time, 5.0)
```

## 部署指南

### 本地部署

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置环境**
```bash
cp config.example.json config.json
# 编辑配置文件
```

3. **启动服务**
```bash
python 数学格式修复Web界面.py
```

### 生产部署

1. **使用Docker**
```bash
docker build -t math-format-fix .
docker run -p 5000:5000 math-format-fix
```

2. **使用Docker Compose**
```bash
docker-compose up -d
```

3. **使用Kubernetes**
```bash
kubectl apply -f k8s/
```

## 贡献指南

### 贡献类型

- 🐛 Bug修复
- ✨ 新功能
- 📚 文档改进
- 🧪 测试用例
- 🔧 工具改进

### 贡献流程

1. **Fork项目**
2. **创建分支**
3. **提交更改**
4. **创建Pull Request**

### 代码审查

- 代码质量检查
- 功能测试验证
- 文档更新确认
- 性能影响评估

## 发布流程

### 版本管理

使用语义化版本号：`主版本.次版本.修订版本`

- 主版本：不兼容的API修改
- 次版本：向下兼容的功能性新增
- 修订版本：向下兼容的问题修正

### 发布步骤

1. **更新版本号**
```bash
# 更新__version__.py
echo '__version__ = "1.0.1"' > __version__.py
```

2. **更新文档**
- 更新README
- 更新CHANGELOG
- 更新API文档

3. **创建标签**
```bash
git tag v1.0.1
git push origin v1.0.1
```

4. **发布到PyPI**
```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

## 维护指南

### 日常维护

- 定期更新依赖
- 监控系统性能
- 处理用户反馈
- 修复安全问题

### 性能优化

- 代码性能分析
- 内存使用优化
- 并发处理优化
- 缓存策略优化

### 安全维护

- 依赖安全扫描
- 代码安全审查
- 漏洞修复
- 安全更新

---

*开发指南生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"开发指南已生成: {output_path}")
    
    def generate_all_docs(self):
        """生成所有文档"""
        print("开始生成项目文档...")
        
        # 创建文档目录
        docs_dir = Path('docs')
        docs_dir.mkdir(exist_ok=True)
        
        # 生成各种文档
        self.generate_readme(docs_dir / 'README.md')
        self.generate_api_docs(docs_dir / 'API文档.md')
        self.generate_user_manual(docs_dir / '用户手册.md')
        self.generate_development_guide(docs_dir / '开发指南.md')
        
        # 生成文档索引
        self.generate_docs_index(docs_dir / '文档索引.md')
        
        print("所有文档生成完成！")
    
    def generate_docs_index(self, output_path: str):
        """生成文档索引"""
        content = f"""# 数学格式修复项目文档索引

## 文档概览

本项目提供完整的文档体系，包括用户文档、开发文档和API文档。

## 文档列表

### 用户文档

| 文档名称 | 描述 | 适用对象 |
|----------|------|----------|
| [README.md](README.md) | 项目介绍和快速开始 | 所有用户 |
| [用户手册.md](用户手册.md) | 详细使用说明 | 最终用户 |
| [API文档.md](API文档.md) | API接口说明 | 开发者 |

### 开发文档

| 文档名称 | 描述 | 适用对象 |
|----------|------|----------|
| [开发指南.md](开发指南.md) | 开发环境搭建和规范 | 开发者 |
| [项目规范.md](../数学格式修复项目完整规范总结文档.md) | 项目开发规范 | 开发者 |
| [测试指南.md](../数学格式修复测试套件.py) | 测试用例和规范 | 测试人员 |

### 项目文档

| 文档名称 | 描述 | 适用对象 |
|----------|------|----------|
| [项目总结.md](../数学格式修复项目最终总结文档.md) | 项目成果总结 | 项目管理者 |
| [项目索引.md](../数学格式修复项目完整索引文档.md) | 项目完整索引 | 项目参与者 |

## 文档结构

```
docs/
├── README.md              # 项目介绍
├── API文档.md             # API接口文档
├── 用户手册.md             # 用户使用手册
├── 开发指南.md             # 开发指南
└── 文档索引.md             # 本文档
```

## 文档更新

- 文档版本与项目版本保持一致
- 重要功能更新时同步更新文档
- 定期检查和更新文档内容
- 收集用户反馈改进文档

## 文档规范

### 格式规范

- 使用Markdown格式
- 中文文档使用UTF-8编码
- 图片使用相对路径
- 代码块标注语言类型

### 内容规范

- 结构清晰，层次分明
- 内容准确，示例完整
- 语言简洁，易于理解
- 及时更新，保持同步

---

*文档索引生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"文档索引已生成: {output_path}")

def main():
    """主函数"""
    generator = DocumentationGenerator()
    generator.generate_all_docs()

if __name__ == '__main__':
    main() 