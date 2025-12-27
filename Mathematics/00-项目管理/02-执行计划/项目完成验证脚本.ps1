# 项目完成验证脚本
# 用于全面验证项目是否达到100%完成标准

param(
    [string]$TargetPath = ".",
    [switch]$Verbose = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Mathematics项目完成验证系统" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$results = @{
    ContentIntegrity = @{ Passed = 0; Failed = 0; Total = 0 }
    QualityStandard = @{ Passed = 0; Failed = 0; Total = 0 }
    LinkValidity = @{ Passed = 0; Failed = 0; Total = 0 }
    StructureIntegrity = @{ Passed = 0; Failed = 0; Total = 0 }
}

$issues = @()

# Phase 1: 内容完整性验证
Write-Host "Phase 1: 内容完整性验证..." -ForegroundColor Yellow
Write-Host ""

# 检查目标目录结构
$targetDirs = @(
    "Mathematics\01-核心内容\Analysis\01-基础理论",
    "Mathematics\01-核心内容\Analysis\02-核心定理",
    "Mathematics\01-核心内容\Analysis\03-应用实例",
    "Mathematics\01-核心内容\Refactor\01-重构理论",
    "Mathematics\01-核心内容\Refactor\02-形式化方法",
    "Mathematics\01-核心内容\Refactor\03-工具与实现",
    "Mathematics\01-核心内容\Lean\01-核心代码",
    "Mathematics\01-核心内容\Lean\02-练习与示例",
    "Mathematics\01-核心内容\Lean\03-文档与说明",
    "Mathematics\01-核心内容\Matter\01-核心内容",
    "Mathematics\01-核心内容\Matter\02-视图文件",
    "Mathematics\01-核心内容\Matter\03-文档"
)

$allDirsExist = $true
foreach ($dir in $targetDirs) {
    $results.ContentIntegrity.Total++
    if (Test-Path $dir) {
        $results.ContentIntegrity.Passed++
        if ($Verbose) {
            Write-Host "  ✓ $dir" -ForegroundColor Green
        }
    } else {
        $results.ContentIntegrity.Failed++
        $allDirsExist = $false
        $issues += "目录不存在: $dir"
        Write-Host "  ✗ $dir (缺失)" -ForegroundColor Red
    }
}

# 检查文件数量
$targetBase = "Mathematics\01-核心内容"
if (Test-Path $targetBase) {
    $fileCount = (Get-ChildItem -Path $targetBase -Recurse -File | Measure-Object).Count
    Write-Host "  目标目录文件数: $fileCount" -ForegroundColor $(if ($fileCount -gt 0) { "Green" } else { "Yellow" })
    if ($fileCount -eq 0) {
        $issues += "目标目录为空，内容尚未迁移"
    }
} else {
    $issues += "目标目录不存在: $targetBase"
}

Write-Host ""

# Phase 2: 质量达标验证
Write-Host "Phase 2: 质量达标验证..." -ForegroundColor Yellow
Write-Host ""

# 检查Lean模块sorry占位符
$leanFiles = Get-ChildItem -Path "Math\Lean" -Filter "*.lean" -Recurse -ErrorAction SilentlyContinue
$sorryCount = 0
if ($leanFiles) {
    foreach ($file in $leanFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content -match 'sorry') {
            $sorryCount += ([regex]::Matches($content, 'sorry')).Count
        }
    }
}

$results.QualityStandard.Total++
if ($sorryCount -eq 0) {
    $results.QualityStandard.Passed++
    Write-Host "  ✓ Lean模块无sorry占位符" -ForegroundColor Green
} else {
    $results.QualityStandard.Failed++
    $issues += "Lean模块仍有 $sorryCount 个sorry占位符未修复"
    Write-Host "  ✗ Lean模块仍有 $sorryCount 个sorry占位符" -ForegroundColor Red
}

# 检查README完成度声明
$readmeFiles = @(
    "README.md",
    "Math\README.md",
    "Math\Analysis\README.md",
    "Math\Refactor\README.md",
    "Math\Lean\README.md",
    "Math\Matter\README.md"
)

$readmeIssues = 0
foreach ($readme in $readmeFiles) {
    if (Test-Path $readme) {
        $content = Get-Content $readme -Raw -ErrorAction SilentlyContinue
        if ($content -match '100%\s*完成|100%\s*完整') {
            $readmeIssues++
            $issues += "README包含虚假完成声明: $readme"
        }
    }
}

$results.QualityStandard.Total++
if ($readmeIssues -eq 0) {
    $results.QualityStandard.Passed++
    Write-Host "  ✓ 所有README完成度声明准确" -ForegroundColor Green
} else {
    $results.QualityStandard.Failed++
    Write-Host "  ✗ 发现 $readmeIssues 个README包含虚假完成声明" -ForegroundColor Red
}

Write-Host ""

# Phase 3: 链接有效性验证
Write-Host "Phase 3: 链接有效性验证..." -ForegroundColor Yellow
Write-Host ""

# 使用链接检查工具
$mdFiles = Get-ChildItem -Path $TargetPath -Filter "*.md" -Recurse | Where-Object { 
    $_.FullName -notlike "*\node_modules\*" -and 
    $_.FullName -notlike "*\.git\*" -and
    $_.FullName -notlike "*\Backup\*"
}

$linkCount = 0
$brokenLinks = 0

foreach ($file in $mdFiles) {
    $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
    if ($null -eq $content) { continue }
    
    $fileDir = Split-Path $file.FullName -Parent
    $matches = [regex]::Matches($content, '\[([^\]]+)\]\(([^\)]+)\)')
    
    foreach ($match in $matches) {
        $linkCount++
        $linkPath = $match.Groups[2].Value
        
        # 跳过外部链接
        if ($linkPath -match '^https?://') { continue }
        
        # 处理相对路径
        $targetPath = Join-Path $fileDir $linkPath
        
        # 处理锚点链接
        if ($linkPath -match '^(.+)#(.+)$') {
            $targetPath = Join-Path $fileDir $matches[1]
        }
        
        # 检查文件是否存在
        if (-not (Test-Path $targetPath)) {
            $brokenLinks++
            $results.LinkValidity.Total++
            $results.LinkValidity.Failed++
            if ($brokenLinks -le 10) {  # 只显示前10个
                $issues += "断链: $($file.FullName) -> $linkPath"
            }
        } else {
            $results.LinkValidity.Total++
            $results.LinkValidity.Passed++
        }
    }
}

Write-Host "  检查了 $linkCount 个链接" -ForegroundColor White
if ($brokenLinks -eq 0) {
    Write-Host "  ✓ 所有链接有效" -ForegroundColor Green
} else {
    Write-Host "  ✗ 发现 $brokenLinks 个断链" -ForegroundColor Red
}

Write-Host ""

# Phase 4: 结构完整性验证
Write-Host "Phase 4: 结构完整性验证..." -ForegroundColor Yellow
Write-Host ""

# 检查关键文档
$keyDocs = @(
    "Mathematics\00-项目管理\02-执行计划\全面任务编排与推进计划.md",
    "Mathematics\00-项目管理\02-执行计划\项目全面完成推进路线图.md",
    "Mathematics\00-项目管理\04-状态跟踪\PROJECT_STATUS.md",
    "Mathematics\02-标准与规范\01-质量标准.md",
    "Mathematics\02-标准与规范\02-质量检查清单.md"
)

$allDocsExist = $true
foreach ($doc in $keyDocs) {
    $results.StructureIntegrity.Total++
    if (Test-Path $doc) {
        $results.StructureIntegrity.Passed++
        if ($Verbose) {
            Write-Host "  ✓ $doc" -ForegroundColor Green
        }
    } else {
        $results.StructureIntegrity.Failed++
        $allDocsExist = $false
        $issues += "关键文档缺失: $doc"
        Write-Host "  ✗ $doc (缺失)" -ForegroundColor Red
    }
}

Write-Host ""

# 输出结果
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "验证结果" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$totalPassed = 0
$totalFailed = 0
$totalTotal = 0

foreach ($category in $results.Keys) {
    $passed = $results[$category].Passed
    $failed = $results[$category].Failed
    $total = $results[$category].Total
    
    $totalPassed += $passed
    $totalFailed += $failed
    $totalTotal += $total
    
    $percentage = if ($total -gt 0) { [math]::Round(($passed / $total) * 100, 1) } else { 0 }
    $status = if ($failed -eq 0) { "✓" } else { "✗" }
    
    Write-Host "$status $category : $passed/$total ($percentage%)" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
}

Write-Host ""
$overallPercentage = if ($totalTotal -gt 0) { [math]::Round(($totalPassed / $totalTotal) * 100, 1) } else { 0 }
Write-Host "总体通过率: $totalPassed/$totalTotal ($overallPercentage%)" -ForegroundColor $(if ($totalFailed -eq 0) { "Green" } else { "Yellow" })

if ($issues.Count -gt 0) {
    Write-Host ""
    Write-Host "发现的问题 ($($issues.Count)个):" -ForegroundColor Yellow
    foreach ($issue in $issues) {
        Write-Host "  - $issue" -ForegroundColor Yellow
    }
}

Write-Host ""
if ($totalFailed -eq 0) {
    Write-Host "✓ 项目验证通过！已达到100%完成标准！" -ForegroundColor Green
} else {
    Write-Host "⚠ 项目验证未完全通过，需要修复 $totalFailed 个问题" -ForegroundColor Yellow
}

Write-Host ""
