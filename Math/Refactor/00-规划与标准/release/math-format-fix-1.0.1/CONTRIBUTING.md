# 贡献指南

感谢您对数学格式修复项目的关注！我们欢迎所有形式的贡献。

## 如何贡献

### 1. 报告问题
如果您发现了bug或有功能建议，请通过以下方式联系我们：
- 在GitHub上创建Issue
- 发送邮件到 support@math-format-fix.com

### 2. 提交代码
1. Fork项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 3. 代码规范
- 遵循PEP 8代码规范
- 添加适当的注释和文档
- 编写单元测试
- 确保所有测试通过

### 4. 文档贡献
- 改进现有文档
- 添加使用示例
- 翻译文档到其他语言

## 开发环境设置

### 1. 克隆仓库
```bash
git clone https://github.com/math-format-fix/project.git
cd project
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. 运行测试
```bash
pytest
```

### 4. 代码检查
```bash
pylint 数学格式修复*.py
flake8 数学格式修复*.py
```

## 提交规范

我们使用[约定式提交](https://www.conventionalcommits.org/)规范：

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat: 添加新的数学格式支持
fix: 修复LaTeX解析错误
docs: 更新API文档
```

## 行为准则

我们致力于为每个人提供友好、安全和欢迎的环境。请阅读我们的[行为准则](CODE_OF_CONDUCT.md)。

## 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

## 联系方式

- 项目维护者: maintainer@math-format-fix.com
- 技术支持: support@math-format-fix.com
- 项目主页: https://github.com/math-format-fix/project

感谢您的贡献！
