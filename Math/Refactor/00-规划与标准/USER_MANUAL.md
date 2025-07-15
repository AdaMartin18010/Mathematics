# 数学格式修复项目用户手册

## 快速开始

### 安装
```bash
# 从源码安装
git clone https://github.com/math-format-fix/project.git
cd project
pip install -r requirements.txt

# 或使用pip安装
pip install math-format-fix
```

### 基本使用
```bash
# 命令行使用
math-format-fix -f input.md -o output.md

# 批量处理
math-format-fix -d input_directory -o output_directory

# 检查格式
math-format-fix --check input.md
```

## 功能特性

### 1. 格式检查
- 自动识别数学公式格式问题
- 提供详细的错误报告
- 支持多种格式类型

### 2. 格式修复
- 智能修复格式错误
- 保持内容完整性
- 支持自定义修复规则

### 3. 批量处理
- 支持大规模文件处理
- 并行处理提高效率
- 实时进度监控

### 4. 多种界面
- 命令行界面
- Web图形界面
- API服务接口

## 配置说明

### 配置文件
项目使用YAML格式的配置文件，支持以下配置项：

```yaml
# 基本配置
format:
  default: markdown
  supported: [markdown, latex, html]

# 处理选项
processing:
  parallel: true
  max_workers: 4
  timeout: 30

# 输出选项
output:
  backup: true
  report: true
  verbose: true
```

### 自定义规则
支持用户自定义修复规则：

```yaml
rules:
  - name: "fix_superscript"
    pattern: "x\^([0-9]+)"
    replacement: "x^{$1}"
    description: "修复上标格式"
```

## 常见问题

### Q: 如何处理大文件？
A: 项目支持分块处理，建议将大文件分割后处理。

### Q: 支持哪些格式？
A: 目前支持Markdown、LaTeX、HTML格式，更多格式正在开发中。

### Q: 如何自定义修复规则？
A: 可以通过配置文件或API接口添加自定义规则。

### Q: 处理速度如何？
A: 单个文件处理时间通常在1秒以内，批量处理支持并行加速。

## 技术支持

- **文档**: https://math-format-fix.com/docs
- **GitHub**: https://github.com/math-format-fix/project
- **邮箱**: support@math-format-fix.com
- **社区**: https://github.com/math-format-fix/project/discussions
