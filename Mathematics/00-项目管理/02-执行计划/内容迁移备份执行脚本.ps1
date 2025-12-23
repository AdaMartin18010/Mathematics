# 内容迁移备份执行脚本
# Content Migration Backup Execution Script
# 
# 创建日期: 2025-12-20
# 用途: 执行内容迁移前的完整备份
# 
# 使用方法: 在项目根目录执行: .\Mathematics\00-项目管理\02-执行计划\内容迁移备份执行脚本.ps1

# 设置错误处理
$ErrorActionPreference = "Stop"

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $ScriptDir))
$BackupRoot = Join-Path (Split-Path -Parent $ProjectRoot) "Mathematics_Backup_2025-12-20"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "内容迁移备份执行脚本" -ForegroundColor Cyan
Write-Host "Content Migration Backup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 显示配置信息
Write-Host "备份配置:" -ForegroundColor Yellow
Write-Host "  项目根目录: $ProjectRoot" -ForegroundColor Gray
Write-Host "  备份目标目录: $BackupRoot" -ForegroundColor Gray
Write-Host ""

# 确认执行
$confirmation = Read-Host "是否继续执行备份? (Y/N)"
if ($confirmation -ne 'Y' -and $confirmation -ne 'y') {
    Write-Host "备份已取消" -ForegroundColor Yellow
    exit
}

Write-Host ""
Write-Host "开始备份..." -ForegroundColor Green
Write-Host ""

# Step 1: 创建备份目录
Write-Host "[1/5] 创建备份目录..." -ForegroundColor Cyan
try {
    New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $BackupRoot "Math") -Force | Out-Null
    Write-Host "  ✓ 备份目录创建成功" -ForegroundColor Green
} catch {
    Write-Host "  ✗ 备份目录创建失败: $_" -ForegroundColor Red
    exit 1
}

# Step 2: 备份核心模块
Write-Host ""
Write-Host "[2/5] 备份核心模块..." -ForegroundColor Cyan

$modules = @(
    @{Name="Analysis"; Path="Math\Analysis"},
    @{Name="Refactor"; Path="Math\Refactor"},
    @{Name="Lean"; Path="Math\Lean"},
    @{Name="Matter"; Path="Math\Matter"},
    @{Name="AI-Mathematics"; Path="Math\AI-Mathematics-Science-2025"},
    @{Name="Archives"; Path="Math\Archives"}
)

$totalFiles = 0
$totalSize = 0

foreach ($module in $modules) {
    $sourcePath = Join-Path $ProjectRoot $module.Path
    $destPath = Join-Path $BackupRoot $module.Path
    
    if (Test-Path $sourcePath) {
        Write-Host "  备份 $($module.Name)..." -ForegroundColor Gray
        try {
            # 计算文件数和大小
            $files = Get-ChildItem -Path $sourcePath -Recurse -File -ErrorAction SilentlyContinue
            $fileCount = ($files | Measure-Object).Count
            $fileSize = ($files | Measure-Object -Property Length -Sum).Sum / 1MB
            
            # 复制文件
            Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force -ErrorAction Stop
            
            $totalFiles += $fileCount
            $totalSize += $fileSize
            
            Write-Host "    ✓ $($module.Name): $fileCount 个文件, $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
        } catch {
            Write-Host "    ✗ $($module.Name) 备份失败: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "    ⚠ $($module.Name) 目录不存在，跳过" -ForegroundColor Yellow
    }
}

# Step 3: 备份根目录文件
Write-Host ""
Write-Host "[3/5] 备份根目录文件..." -ForegroundColor Cyan
try {
    $rootFiles = Get-ChildItem -Path $ProjectRoot -Filter "*.md" -File -ErrorAction SilentlyContinue
    foreach ($file in $rootFiles) {
        Copy-Item -Path $file.FullName -Destination (Join-Path $BackupRoot $file.Name) -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  ✓ 根目录文件备份完成" -ForegroundColor Green
} catch {
    Write-Host "  ✗ 根目录文件备份失败: $_" -ForegroundColor Red
}

# Step 4: 验证备份完整性
Write-Host ""
Write-Host "[4/5] 验证备份完整性..." -ForegroundColor Cyan

try {
    # 统计源文件数量
    $sourceCount = 0
    foreach ($module in $modules) {
        $sourcePath = Join-Path $ProjectRoot $module.Path
        if (Test-Path $sourcePath) {
            $count = (Get-ChildItem -Path $sourcePath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
            $sourceCount += $count
        }
    }
    
    # 统计备份文件数量
    $backupCount = (Get-ChildItem -Path $BackupRoot -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    
    Write-Host "  源文件数: $sourceCount" -ForegroundColor Gray
    Write-Host "  备份文件数: $backupCount" -ForegroundColor Gray
    
    if ($backupCount -ge $sourceCount * 0.95) {
        Write-Host "  ✓ 备份完整性验证通过（允许5%差异）" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ 备份完整性验证警告：文件数差异较大" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠ 备份完整性验证失败: $_" -ForegroundColor Yellow
}

# Step 5: 创建备份清单
Write-Host ""
Write-Host "[5/5] 创建备份清单..." -ForegroundColor Cyan

try {
    $manifestPath = Join-Path $BackupRoot "备份清单.txt"
    $manifest = @"
内容迁移备份清单
Content Migration Backup Manifest

备份日期: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
备份位置: $BackupRoot

模块统计:
"@
    
    foreach ($module in $modules) {
        $destPath = Join-Path $BackupRoot $module.Path
        if (Test-Path $destPath) {
            $files = Get-ChildItem -Path $destPath -Recurse -File -ErrorAction SilentlyContinue
            $fileCount = ($files | Measure-Object).Count
            $fileSize = ($files | Measure-Object -Property Length -Sum).Sum / 1MB
            $manifest += "`n  $($module.Name): $fileCount 个文件, $([math]::Round($fileSize, 2)) MB"
        }
    }
    
    $manifest += "`n`n总计: $totalFiles 个文件, $([math]::Round($totalSize, 2)) MB"
    $manifest += "`n`n备份完成时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")"
    
    $manifest | Out-File -FilePath $manifestPath -Encoding UTF8
    Write-Host "  ✓ 备份清单已创建: $manifestPath" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ 备份清单创建失败: $_" -ForegroundColor Yellow
}

# 完成
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "备份完成!" -ForegroundColor Green
Write-Host "Backup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "备份位置: $BackupRoot" -ForegroundColor Yellow
Write-Host "备份文件数: $totalFiles" -ForegroundColor Yellow
Write-Host "备份大小: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Yellow
Write-Host ""
Write-Host "下一步: 可以开始内容迁移Phase 2" -ForegroundColor Cyan
Write-Host ""
