# 自动化检查工具脚本
# 用于检查文档格式、链接有效性和数学公式语法

param(
    [string]$CheckType = "all",
    [string]$TargetPath = "."
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Mathematics项目自动化检查工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查类型：all, format, links, math
$checkFormat = ($CheckType -eq "all" -or $CheckType -eq "format")
$checkLinks = ($CheckType -eq "all" -or $CheckType -eq "links")
$checkMath = ($CheckType -eq "all" -or $CheckType -eq "math")

$errors = @()
$warnings = @()
$checkedFiles = 0

# 1. 文档格式检查
if ($checkFormat) {
    Write-Host "1. 检查文档格式..." -ForegroundColor Yellow
    $mdFiles = Get-ChildItem -Path $TargetPath -Filter "*.md" -Recurse | Where-Object { $_.FullName -notlike "*\node_modules\*" -and $_.FullName -notlike "*\.git\*" }
    
    foreach ($file in $mdFiles) {
        $checkedFiles++
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        
        if ($null -eq $content) {
            $warnings += "无法读取文件: $($file.FullName)"
            continue
        }
        
        # 检查Markdown表格格式
        if ($content -match '\|.*\|.*\|' -and $content -notmatch '\|:?-+:?\|') {
            $warnings += "可能的表格格式问题: $($file.FullName)"
        }
        
        # 检查标题层级
        $lines = $content -split "`n"
        $prevLevel = 0
        foreach ($line in $lines) {
            if ($line -match '^(#{1,6})\s+') {
                $level = $matches[1].Length
                if ($level -gt $prevLevel + 1) {
                    $warnings += "标题层级跳跃: $($file.FullName) - $line"
                }
                $prevLevel = $level
            }
        }
    }
    
    Write-Host "   检查了 $checkedFiles 个Markdown文件" -ForegroundColor Green
}

# 2. 链接有效性检查
if ($checkLinks) {
    Write-Host "2. 检查链接有效性..." -ForegroundColor Yellow
    $mdFiles = Get-ChildItem -Path $TargetPath -Filter "*.md" -Recurse | Where-Object { $_.FullName -notlike "*\node_modules\*" -and $_.FullName -notlike "*\.git\*" }
    
    $linkCount = 0
    $brokenLinks = 0
    
    foreach ($file in $mdFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($null -eq $content) { continue }
        
        # 匹配Markdown链接 [text](path)
        $matches = [regex]::Matches($content, '\[([^\]]+)\]\(([^\)]+)\)')
        
        foreach ($match in $matches) {
            $linkCount++
            $linkPath = $match.Groups[2].Value
            
            # 跳过外部链接
            if ($linkPath -match '^https?://') { continue }
            
            # 处理相对路径
            $fileDir = Split-Path $file.FullName -Parent
            $targetPath = Join-Path $fileDir $linkPath
            
            # 处理锚点链接
            if ($linkPath -match '^(.+)#(.+)$') {
                $targetPath = Join-Path $fileDir $matches[1]
            }
            
            # 检查文件是否存在
            if (-not (Test-Path $targetPath)) {
                $brokenLinks++
                $errors += "断链: $($file.FullName) -> $linkPath"
            }
        }
    }
    
    Write-Host "   检查了 $linkCount 个链接，发现 $brokenLinks 个断链" -ForegroundColor $(if ($brokenLinks -eq 0) { "Green" } else { "Red" })
}

# 3. 数学公式语法检查（基础检查）
if ($checkMath) {
    Write-Host "3. 检查数学公式语法..." -ForegroundColor Yellow
    $mdFiles = Get-ChildItem -Path $TargetPath -Filter "*.md" -Recurse | Where-Object { $_.FullName -notlike "*\node_modules\*" -and $_.FullName -notlike "*\.git\*" }
    
    $formulaCount = 0
    $invalidFormulas = 0
    
    foreach ($file in $mdFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($null -eq $content) { continue }
        
        # 匹配LaTeX公式 $...$ 或 $$...$$
        $matches = [regex]::Matches($content, '\$\$?([^\$]+)\$\$?')
        
        foreach ($match in $matches) {
            $formulaCount++
            $formula = $match.Groups[1].Value
            
            # 基础语法检查：括号匹配
            $openBraces = ($formula.ToCharArray() | Where-Object { $_ -eq '{' }).Count
            $closeBraces = ($formula.ToCharArray() | Where-Object { $_ -eq '}' }).Count
            
            if ($openBraces -ne $closeBraces) {
                $invalidFormulas++
                $warnings += "括号不匹配: $($file.FullName) - $formula"
            }
        }
    }
    
    Write-Host "   检查了 $formulaCount 个数学公式，发现 $invalidFormulas 个潜在问题" -ForegroundColor $(if ($invalidFormulas -eq 0) { "Green" } else { "Yellow" })
}

# 输出结果
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "检查完成" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($errors.Count -gt 0) {
    Write-Host "错误 ($($errors.Count)个):" -ForegroundColor Red
    foreach ($error in $errors) {
        Write-Host "  - $error" -ForegroundColor Red
    }
    Write-Host ""
}

if ($warnings.Count -gt 0) {
    Write-Host "警告 ($($warnings.Count)个):" -ForegroundColor Yellow
    foreach ($warning in $warnings) {
        Write-Host "  - $warning" -ForegroundColor Yellow
    }
    Write-Host ""
}

if ($errors.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host "✓ 所有检查通过！" -ForegroundColor Green
}

Write-Host ""
Write-Host "使用方法:" -ForegroundColor Cyan
Write-Host "  .\自动化检查工具脚本.ps1 -CheckType all        # 检查所有"
Write-Host "  .\自动化检查工具脚本.ps1 -CheckType format     # 只检查格式"
Write-Host "  .\自动化检查工具脚本.ps1 -CheckType links      # 只检查链接"
Write-Host "  .\自动化检查工具脚本.ps1 -CheckType math       # 只检查数学公式"
Write-Host ""
