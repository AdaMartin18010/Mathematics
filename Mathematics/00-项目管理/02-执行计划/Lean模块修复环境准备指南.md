# Lean模块修复环境准备指南

**创建日期**: 2025-12-20
**状态**: 准备中
**优先级**: P0
**目标**: 准备Lean模块修复所需的环境和工具

---

## 📋 环境要求

### 必需软件

1. **Lean 4**
   - 版本: 最新稳定版
   - 下载: https://leanprover-community.github.io/get_started.html
   - 安装: 按照官方文档安装

2. **Mathlib 4**
   - 版本: 最新版本
   - 安装: 使用`lake`工具安装
   - 文档: https://leanprover-community.github.io/mathlib4_docs/

3. **编辑器**
   - VS Code + Lean 4扩展（推荐）
   - 或 Emacs + lean4-mode

---

## 🚀 环境设置步骤

### Step 1: 安装Lean 4

#### Windows

1. 下载Lean 4安装器
2. 运行安装器
3. 验证安装：
   ```bash
   lean --version
   ```

#### Linux/Mac

```bash
# 使用elan安装
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### Step 2: 安装Mathlib 4

```bash
# 进入Lean项目目录
cd Math/Lean

# 初始化lake项目（如果还没有）
lake init

# 添加mathlib依赖
lake exe cache get
```

### Step 3: 配置编辑器

#### VS Code

1. 安装Lean 4扩展
2. 打开项目目录
3. 等待Lean服务器启动
4. 验证：打开任意.lean文件，应看到类型信息

---

## 🔧 项目配置

### 检查项目结构

```
Math/Lean/
├── Exercises/
│   ├── Analysis/
│   │   └── Real.lean  # 需要修复的文件
│   ├── Topology/
│   │   └── Basic.lean  # 需要修复的文件
│   └── ...
└── lakefile.lean  # 项目配置文件
```

### 验证环境

1. **编译测试**
   ```bash
   cd Math/Lean
   lake build
   ```

2. **类型检查**
   - 打开`Exercises/Analysis/Real.lean`
   - 检查是否有错误
   - 确认可以显示类型信息

---

## 📝 修复工作流程

### 标准修复流程

1. **定位sorry**
   - 使用搜索功能找到`sorry`
   - 查看上下文和类型签名

2. **分析问题**
   - 理解需要证明的定理
   - 查看mathlib4相关API
   - 确定证明策略

3. **编写证明**
   - 使用Lean 4语法
   - 参考mathlib4文档
   - 逐步构建证明

4. **验证修复**
   - 编译检查
   - 类型检查
   - 逻辑验证

---

## 🔍 工具和资源

### 在线资源

1. **Mathlib 4文档**
   - https://leanprover-community.github.io/mathlib4_docs/
   - 搜索相关定理和API

2. **Lean 4手册**
   - https://leanprover.github.io/lean4/doc/
   - 语法和功能说明

3. **Zulip社区**
   - https://leanprover.zulipchat.com/
   - 提问和讨论

### 本地工具

1. **Lean Infoview**
   - VS Code中显示类型和证明状态
   - 快捷键: `Ctrl+Shift+Enter`

2. **Goal View**
   - 显示当前证明目标
   - 帮助理解证明状态

---

## ✅ 环境验证清单

### 安装验证

- [ ] Lean 4已安装并可运行
- [ ] Mathlib 4已安装
- [ ] 编辑器已配置
- [ ] 项目可以编译

### 功能验证

- [ ] 可以打开.lean文件
- [ ] 可以显示类型信息
- [ ] 可以编译项目
- [ ] 可以搜索mathlib4文档

---

## 🎯 准备完成标志

环境准备完成的标准：

1. ✅ Lean 4可以运行
2. ✅ Mathlib 4已安装
3. ✅ 项目可以编译
4. ✅ 编辑器正常工作
5. ✅ 可以查看类型信息

---

## 🔗 相关文档

- [Lean模块修复执行计划](./Lean模块修复执行计划.md)
- [Lean模块第一批修复清单](./Lean模块第一批修复清单.md)
- [全面任务编排与推进计划](./全面任务编排与推进计划.md)

---

**更新频率**: 每次环境变更后更新
**最后更新**: 2025-12-20
**维护**: Mathematics项目团队
