# 内容迁移Phase 3-4执行指南

**创建日期**: 2025-12-20
**版本**: v1.0
**状态**: 准备执行
**优先级**: P1

---

## 📋 执行概述

Phase 3-4将迁移Lean和Matter模块的核心内容到新的目录结构，并更新所有交叉引用。

---

## ✅ 前置条件

在执行Phase 3-4之前，必须完成：

- [x] Phase 1准备完成（90%）
- [ ] Phase 2执行完成（Analysis和Refactor模块已迁移）
- [ ] 验证Phase 2迁移完整性
- [ ] 所有Phase 2链接已更新

---

## 🎯 执行目标

1. **迁移Lean模块核心内容**
   - 迁移核心Lean代码文件
   - 迁移Lean文档
   - 保留重要示例

2. **迁移Matter模块核心内容**
   - 迁移核心Matter文档
   - 迁移重要视图文件
   - 保留关键内容

3. **更新所有交叉引用**
   - 修复模块间链接
   - 更新文档引用
   - 验证链接有效性

---

## 📝 详细执行步骤

### Phase 3: 迁移Lean模块

**时间**: 3-4小时

#### Step 1: 准备Lean模块目标目录

**时间**: 10分钟

**步骤**:

1. 创建目标目录结构：

   ```powershell
   New-Item -ItemType Directory -Path "Mathematics\01-核心内容\Lean\01-核心代码" -Force
   New-Item -ItemType Directory -Path "Mathematics\01-核心内容\Lean\02-练习与示例" -Force
   New-Item -ItemType Directory -Path "Mathematics\01-核心内容\Lean\03-文档与说明" -Force
   ```

2. 验证目录结构

**验收标准**:

- ✅ 所有目标目录已创建
- ✅ 目录结构正确

---

#### Step 2: 迁移核心Lean代码

**时间**: 1-2小时

**步骤**:

1. **识别核心代码文件**（参考 `Lean模块修复执行计划.md`）
   - 核心定义文件（10-15个）
   - 重要定理文件（15-20个）
   - 关键示例文件（5-10个）

2. **迁移核心定义文件**

   ```powershell
   # 示例：迁移核心定义文件
   Copy-Item "Math\Lean\Exercises\Analysis\*.lean" -Destination "Mathematics\01-核心内容\Lean\01-核心代码\Analysis\" -Recurse
   ```

3. **迁移重要定理文件**

   ```powershell
   # 示例：迁移重要定理文件
   Copy-Item "Math\Lean\Exercises\Topology\*.lean" -Destination "Mathematics\01-核心内容\Lean\01-核心代码\Topology\" -Recurse
   ```

4. **迁移关键示例文件**

   ```powershell
   # 示例：迁移关键示例文件
   Copy-Item "Math\Lean\Exercises\Examples\*.lean" -Destination "Mathematics\01-核心内容\Lean\02-练习与示例\" -Recurse
   ```

5. **验证迁移结果**
   - 检查文件数量
   - 随机抽样检查文件内容
   - 确认文件完整性

**验收标准**:

- ✅ 核心代码文件已迁移
- ✅ 文件数量合理
- ✅ 文件内容完整
- ✅ 无文件损坏

---

#### Step 3: 迁移Lean文档

**时间**: 30分钟

**步骤**:

1. **识别核心文档**
   - README文件
   - 导航文档
   - 重要说明文档

2. **迁移文档**

   ```powershell
   # 示例：迁移Lean文档
   Copy-Item "Math\Lean\README.md" -Destination "Mathematics\01-核心内容\Lean\03-文档与说明\README.md"
   Copy-Item "Math\Lean\*.md" -Destination "Mathematics\01-核心内容\Lean\03-文档与说明\" -Recurse
   ```

3. **验证迁移结果**

**验收标准**:

- ✅ 核心文档已迁移
- ✅ 文档内容完整

---

### Phase 4: 迁移Matter模块

**时间**: 2-3小时

#### Step 1: 准备Matter模块目标目录

**时间**: 10分钟

**步骤**:

1. 创建目标目录结构：

   ```powershell
   New-Item -ItemType Directory -Path "Mathematics\01-核心内容\Matter\01-核心内容" -Force
   New-Item -ItemType Directory -Path "Mathematics\01-核心内容\Matter\02-视图文件" -Force
   New-Item -ItemType Directory -Path "Mathematics\01-核心内容\Matter\03-文档" -Force
   ```

2. 验证目录结构

**验收标准**:

- ✅ 所有目标目录已创建
- ✅ 目录结构正确

---

#### Step 2: 迁移Matter核心内容

**时间**: 1-2小时

**步骤**:

1. **识别核心内容**（参考 `Matter模块优化执行计划.md`）
   - 核心文档（10-15个）
   - 重要视图文件（20-30个）
   - 关键配置文件（5-10个）

2. **迁移核心文档**

   ```powershell
   # 示例：迁移核心文档
   Copy-Item "Math\Matter\*.md" -Destination "Mathematics\01-核心内容\Matter\01-核心内容\" -Recurse -Exclude "Archives\*"
   ```

3. **迁移重要视图文件**

   ```powershell
   # 示例：迁移重要视图文件
   Copy-Item "Math\Matter\views\*.md" -Destination "Mathematics\01-核心内容\Matter\02-视图文件\" -Recurse
   ```

4. **验证迁移结果**

**验收标准**:

- ✅ 核心内容已迁移
- ✅ 文件数量合理
- ✅ 文件内容完整

---

#### Step 3: 迁移Matter文档

**时间**: 30分钟

**步骤**:

1. **迁移README和文档**

   ```powershell
   # 示例：迁移Matter文档
   Copy-Item "Math\Matter\README.md" -Destination "Mathematics\01-核心内容\Matter\03-文档\README.md"
   ```

2. **验证迁移结果**

**验收标准**:

- ✅ 文档已迁移
- ✅ 文档内容完整

---

### 更新所有交叉引用

**时间**: 1-2小时

#### Step 1: 使用链接检查工具

**时间**: 15分钟

**步骤**:

1. 运行链接检查工具：

   ```powershell
   .\链接检查与修复脚本.ps1 -TargetPath "Mathematics\01-核心内容"
   ```

2. 查看断链报告

**验收标准**:

- ✅ 链接检查完成
- ✅ 断链已识别

---

#### Step 2: 修复模块间链接

**时间**: 30-60分钟

**步骤**:

1. **修复Analysis模块链接**
   - 更新到Refactor模块的链接
   - 更新到Lean模块的链接
   - 更新到Matter模块的链接

2. **修复Refactor模块链接**
   - 更新到Analysis模块的链接
   - 更新到Lean模块的链接
   - 更新到Matter模块的链接

3. **修复Lean模块链接**
   - 更新到Analysis模块的链接
   - 更新到Refactor模块的链接
   - 更新到Matter模块的链接

4. **修复Matter模块链接**
   - 更新到Analysis模块的链接
   - 更新到Refactor模块的链接
   - 更新到Lean模块的链接

**验收标准**:

- ✅ 所有模块间链接已更新
- ✅ 链接路径正确

---

#### Step 3: 使用自动修复工具

**时间**: 15分钟

**步骤**:

1. 运行自动修复工具：

   ```powershell
   .\链接检查与修复脚本.ps1 -TargetPath "Mathematics\01-核心内容" -Fix
   ```

2. 验证修复结果

**验收标准**:

- ✅ 自动修复完成
- ✅ 修复结果验证通过

---

#### Step 4: 验证链接有效性

**时间**: 15分钟

**步骤**:

1. 再次运行链接检查工具：

   ```powershell
   .\链接检查与修复脚本.ps1 -TargetPath "Mathematics\01-核心内容"
   ```

2. 确认无断链

**验收标准**:

- ✅ 所有链接有效
- ✅ 无断链

---

### 验证迁移完整性

**时间**: 30分钟

#### Step 1: 文件完整性检查

**时间**: 15分钟

**步骤**:

1. 统计迁移文件数量
2. 对比原始文件数量
3. 随机抽样检查文件内容

**验收标准**:

- ✅ 文件数量合理
- ✅ 文件内容完整

---

#### Step 2: 链接完整性检查

**时间**: 15分钟

**步骤**:

1. 运行链接检查工具
2. 确认无断链
3. 验证交叉引用正确

**验收标准**:

- ✅ 所有链接有效
- ✅ 交叉引用正确

---

## 📊 预期结果

### 文件统计

| 模块 | 迁移文件数 | 目标目录 |
|------|----------|---------|
| Lean | 30-50个 | `Mathematics/01-核心内容/Lean/` |
| Matter | 40-60个 | `Mathematics/01-核心内容/Matter/` |
| **总计** | **70-110个** | - |

### 质量指标

- **文件完整性**: 100%
- **链接有效性**: 100%
- **内容准确性**: 保持原样

---

## ⚠️ 风险与应对

### 风险1: Lean代码编译问题

**应对**:

- 保留原始文件
- 验证编译通过
- 逐步迁移

### 风险2: 链接路径复杂

**应对**:

- 使用自动修复工具
- 手动验证关键链接
- 建立链接映射表

### 风险3: 内容丢失

**应对**:

- 完整备份已创建
- 验证迁移完整性
- 保留原始文件（暂时）

---

## 🔗 相关文档

- [内容迁移Phase2执行指南](./内容迁移Phase2执行指南.md)
- [内容迁移备份计划](./内容迁移备份计划.md)
- [内容迁移风险评估报告](./内容迁移风险评估报告.md)
- [Lean模块修复执行计划](./Lean模块修复执行计划.md)
- [Matter模块优化执行计划](./Matter模块优化执行计划.md)
- [链接检查与修复脚本](./链接检查与修复脚本.ps1)

---

## 📅 执行时间表

| 步骤 | 预计时间 | 累计时间 |
|------|---------|---------|
| Phase 3: 迁移Lean模块 | 3-4小时 | 3-4小时 |
| Phase 4: 迁移Matter模块 | 2-3小时 | 5-7小时 |
| 更新交叉引用 | 1-2小时 | 6-9小时 |
| 验证迁移完整性 | 30分钟 | 6.5-9.5小时 |
| **总计** | **6.5-9.5小时** | - |

---

**更新频率**: 根据执行情况更新
**最后更新**: 2025-12-20
**维护**: Mathematics项目团队
