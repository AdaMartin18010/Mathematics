# 链接检查与修复脚本
# 用于检查文档中的链接有效性，并自动修复常见问题

param(
    [string]$TargetPath = ".",
    [switch]$Fix = $false,
    [switch]$Verbose = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Mathematics项目链接检查与修复工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$mdFiles = Get-ChildItem -Path $TargetPath -Filter "*.md" -Recurse | Where-Object { 
    $_.FullName -notlike "*\node_modules\*" -and 
    $_.FullName -notlike "*\.git\*" -and
    $_.FullName -notlike "*\Backup\*"
}

$totalLinks = 0
$brokenLinks = 0
$fixedLinks = 0
$externalLinks = 0
$linkIssues = @()

Write-Host "正在检查 $($mdFiles.Count) 个Markdown文件..." -ForegroundColor Yellow
Write-Host ""

foreach ($file in $mdFiles) {
    $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
    if ($null -eq $content) { continue }
    
    $fileDir = Split-Path $file.FullName -Parent
    $relativePath = $file.FullName.Replace((Resolve-Path $TargetPath).Path + "\", "")
    
    # 匹配Markdown链接 [text](path)
    $matches = [regex]::Matches($content, '\[([^\]]+)\]\(([^\)]+)\)')
    
    foreach ($match in $matches) {
        $totalLinks++
        $linkText = $match.Groups[1].Value
        $linkPath = $match.Groups[2].Value
        
        # 跳过外部链接
        if ($linkPath -match '^https?://') {
            $externalLinks++
            continue
        }
        
        # 处理锚点链接
        $anchor = $null
        $targetPath = $linkPath
        if ($linkPath -match '^(.+)#(.+)$') {
            $targetPath = $matches[1]
            $anchor = $matches[2]
        }
        
        # 处理相对路径
        if (-not [System.IO.Path]::IsPathRooted($targetPath)) {
            $targetPath = Join-Path $fileDir $targetPath
        }
        
        # 规范化路径
        try {
            $targetPath = [System.IO.Path]::GetFullPath($targetPath)
        } catch {
            $brokenLinks++
            $linkIssues += [PSCustomObject]@{
                File = $relativePath
                Link = $linkPath
                Issue = "无效路径"
                Line = ($content.Substring(0, $match.Index) -split "`n").Count
            }
            continue
        }
        
        # 检查文件是否存在
        if (-not (Test-Path $targetPath)) {
            $brokenLinks++
            $linkIssues += [PSCustomObject]@{
                File = $relativePath
                Link = $linkPath
                Issue = "文件不存在"
                Line = ($content.Substring(0, $match.Index) -split "`n").Count
            }
            
            # 尝试自动修复
            if ($Fix) {
                # 尝试修复常见问题
                $fixed = $false
                
                # 修复1: 路径大小写问题
                $parentDir = Split-Path $targetPath -Parent
                if (Test-Path $parentDir) {
                    $fileName = Split-Path $targetPath -Leaf
                    $actualFile = Get-ChildItem -Path $parentDir -Filter $fileName -ErrorAction SilentlyContinue | 
                        Where-Object { $_.Name -eq $fileName -or $_.Name -eq $fileName.ToLower() -or $_.Name -eq $fileName.ToUpper() } | 
                        Select-Object -First 1
                    
                    if ($actualFile) {
                        $newPath = $actualFile.FullName
                        $relativeNewPath = $newPath.Replace((Resolve-Path $TargetPath).Path + "\", "")
                        $relativeToFile = [System.IO.Path]::GetRelativePath($fileDir, $newPath)
                        
                        $content = $content.Replace($match.Value, "[$linkText]($relativeToPath)")
                        $fixed = $true
                        $fixedLinks++
                    }
                }
                
                # 修复2: .md扩展名缺失
                if (-not $fixed -and -not $targetPath.EndsWith(".md")) {
                    $targetPathWithMd = $targetPath + ".md"
                    if (Test-Path $targetPathWithMd) {
                        $relativeToFile = [System.IO.Path]::GetRelativePath($fileDir, $targetPathWithMd)
                        $content = $content.Replace($match.Value, "[$linkText]($relativeToFile)")
                        $fixed = $true
                        $fixedLinks++
                    }
                }
            }
        }
    }
    
    # 如果修复了链接，保存文件
    if ($Fix -and $fixedLinks -gt 0) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
    }
}

# 输出结果
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "检查完成" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "统计信息:" -ForegroundColor Yellow
Write-Host "  总链接数: $totalLinks" -ForegroundColor White
Write-Host "  外部链接: $externalLinks" -ForegroundColor Green
Write-Host "  断链数: $brokenLinks" -ForegroundColor $(if ($brokenLinks -eq 0) { "Green" } else { "Red" })
if ($Fix) {
    Write-Host "  修复链接数: $fixedLinks" -ForegroundColor $(if ($fixedLinks -gt 0) { "Green" } else { "Yellow" })
}
Write-Host ""

if ($brokenLinks -gt 0) {
    Write-Host "断链详情:" -ForegroundColor Red
    foreach ($issue in $linkIssues) {
        Write-Host "  - $($issue.File):$($issue.Line) -> $($issue.Link) ($($issue.Issue))" -ForegroundColor Red
    }
    Write-Host ""
}

if ($brokenLinks -eq 0) {
    Write-Host "✓ 所有链接有效！" -ForegroundColor Green
} elseif ($Fix -and $fixedLinks -gt 0) {
    Write-Host "✓ 已修复 $fixedLinks 个链接" -ForegroundColor Green
    Write-Host "⚠ 仍有 $($brokenLinks - $fixedLinks) 个链接需要手动修复" -ForegroundColor Yellow
} else {
    Write-Host "⚠ 发现 $brokenLinks 个断链，需要手动修复" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "提示: 使用 -Fix 参数可以尝试自动修复常见问题" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "使用方法:" -ForegroundColor Cyan
Write-Host "  .\链接检查与修复脚本.ps1                    # 只检查"
Write-Host "  .\链接检查与修复脚本.ps1 -Fix                # 检查并修复"
Write-Host "  .\链接检查与修复脚本.ps1 -TargetPath '.\Math' # 检查指定目录"
Write-Host ""
