# 创建内容迁移目标目录结构脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "创建内容迁移目标目录结构" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$basePath = "Mathematics\01-核心内容"

# 创建Analysis模块目录
Write-Host "创建Analysis模块目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "$basePath\Analysis\01-基础理论" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\Analysis\02-核心定理" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\Analysis\03-应用实例" -Force | Out-Null
Write-Host "  ✓ Analysis模块目录已创建" -ForegroundColor Green

# 创建Refactor模块目录
Write-Host "创建Refactor模块目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "$basePath\Refactor\01-重构理论" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\Refactor\02-形式化方法" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\Refactor\03-工具与实现" -Force | Out-Null
Write-Host "  ✓ Refactor模块目录已创建" -ForegroundColor Green

# 创建Lean模块目录
Write-Host "创建Lean模块目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "$basePath\Lean\01-核心代码" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\Lean\02-练习与示例" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\Lean\03-文档与说明" -Force | Out-Null
Write-Host "  ✓ Lean模块目录已创建" -ForegroundColor Green

# 创建Matter模块目录
Write-Host "创建Matter模块目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "$basePath\Matter\01-核心内容" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\Matter\02-视图文件" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\Matter\03-文档" -Force | Out-Null
Write-Host "  ✓ Matter模块目录已创建" -ForegroundColor Green

# 创建AI-Mathematics模块目录
Write-Host "创建AI-Mathematics模块目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "$basePath\AI-Mathematics\01-核心理论" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\AI-Mathematics\02-应用实践" -Force | Out-Null
New-Item -ItemType Directory -Path "$basePath\AI-Mathematics\03-文档" -Force | Out-Null
Write-Host "  ✓ AI-Mathematics模块目录已创建" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "目录结构创建完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 验证目录结构
Write-Host "验证目录结构..." -ForegroundColor Yellow
$directories = @(
    "$basePath\Analysis\01-基础理论",
    "$basePath\Analysis\02-核心定理",
    "$basePath\Analysis\03-应用实例",
    "$basePath\Refactor\01-重构理论",
    "$basePath\Refactor\02-形式化方法",
    "$basePath\Refactor\03-工具与实现",
    "$basePath\Lean\01-核心代码",
    "$basePath\Lean\02-练习与示例",
    "$basePath\Lean\03-文档与说明",
    "$basePath\Matter\01-核心内容",
    "$basePath\Matter\02-视图文件",
    "$basePath\Matter\03-文档",
    "$basePath\AI-Mathematics\01-核心理论",
    "$basePath\AI-Mathematics\02-应用实践",
    "$basePath\AI-Mathematics\03-文档"
)

$allExist = $true
foreach ($dir in $directories) {
    if (Test-Path $dir) {
        Write-Host "  ✓ $dir" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $dir (缺失)" -ForegroundColor Red
        $allExist = $false
    }
}

Write-Host ""
if ($allExist) {
    Write-Host "✓ 所有目录结构验证通过！" -ForegroundColor Green
} else {
    Write-Host "⚠ 部分目录结构验证失败，请检查！" -ForegroundColor Yellow
}

Write-Host ""
