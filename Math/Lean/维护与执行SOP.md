# Lean 子项目维护与执行SOP | Maintenance & Execution SOP

## 目的 | Purpose

- 建立标准化的例行维护与发布执行步骤，确保版本同步、构建可用与文档质量持续达标。

## 角色 | Roles

- 维护人（Owner）：负责SOP执行与记录
- 审阅人（Reviewer）：抽检与质量签收

## 周期 | Cadence

- 周例行：快速健康检查（15分钟）
- 月例行：版本同步与构建抽检（半天）

## 周例行巡检 | Weekly Routine (15min)

1) 链接健康抽检（随机10条）
   - 入口：`00-完整导航索引系统.md`（参见“质量门禁与断链检查”）
   - 记录：在 `对标进度表.md` 追加当月“断链抽检记录”
2) 关键文档变更核对（3处）
   - `README.md`、`版本同步索引.md`、本SOP
3) 快速构建烟测（可选）
   - Windows：`cd Exercises && ./build.ps1`

## 月度执行 | Monthly Routine (Half-day)

1) 版本同步
   - 参照：`版本同步索引.md` → 当月记录
   - 动作：检查 Lean4 Releases 与 mathlib4 变更，更新“当月记录”
2) 构建与示例验证
   - Windows：`cd Exercises && ./build.ps1 -Clean`
   - 如失败：按 `Exercises/README.md` 的“构建故障排查”处理，并在 `版本同步索引.md` 回填
3) 文档对齐更新
   - 在以下文件回链/对齐：
     - `README.md`（“快速验证”与外部资源）
     - `00-完整导航索引系统.md`（“构建验证”与“质量门禁”）
     - `对标进度表.md`（审计时间、抽检记录）
4) 质量门禁复核
   - 确认新增/改动文档包含“版本与兼容性注记”或回链到 `版本同步索引.md`

## 产出 | Deliverables

- `版本同步索引.md`：当月记录更新
- `对标进度表.md`：审计时间与抽检记录更新
- `README.md`/`00-完整导航索引系统.md`：必要的链接与验证区更新

## 验证 | Verification

- 构建成功：`Exercises` 全量构建通过
- 链接健康：随机样本0断链
- 文档一致：涉及版本/构建的文档相互回链一致

## 附 | References

- `版本同步索引.md`
- `00-完整导航索引系统.md`
- `Exercises/README.md`
- `对标进度表.md`
