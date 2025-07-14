#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复执行脚本
Math/Refactor项目数学格式全面修复工具

作者: 数学知识体系重构项目组
时间: 2025年1月
版本: 1.0
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

class MathFormatFixer:
    """数学格式修复器"""
    
    def __init__(self, project_path: str):
        """初始化修复器"""
        self.project_path = Path(project_path)
        self.results = {
            'files_processed': 0,
            'errors_fixed': 0,
            'errors_remaining': 0,
            'fix_details': []
        }
        
        # 错误统计
        self.error_stats = {
            'syntax_errors': 0,
            'symbol_errors': 0,
            'format_errors': 0,
            'spacing_errors': 0
        }
    
    def fix_bracket_mismatch(self, text: str) -> str:
        """修复括号不匹配问题"""
        # 查找所有数学公式
        inline_pattern = r'\$[^$]+\$'
        block_pattern = r'\$\$[^$]+\$\$'
        
        def fix_formula(match):
            formula = match.group(0)
            # 计算括号数量
            open_brackets = formula.count('(')
            close_brackets = formula.count(')')
            
            if open_brackets > close_brackets:
                # 添加缺失的右括号
                formula = formula + ')' * (open_brackets - close_brackets)
                self.error_stats['syntax_errors'] += 1
            elif close_brackets > open_brackets:
                # 添加缺失的左括号
                formula = '(' * (close_brackets - open_brackets) + formula
                self.error_stats['syntax_errors'] += 1
            
            return formula
        
        # 修复行内公式
        text = re.sub(inline_pattern, fix_formula, text)
        # 修复块级公式
        text = re.sub(block_pattern, fix_formula, text)
        
        return text
    
    def fix_spacing(self, text: str) -> str:
        """修复数学公式中的空格问题"""
        # 在运算符前后添加空格
        operators = ['+', '-', '=', '\\neq', '\\leq', '\\geq', '\\approx', '\\equiv']
        
        for op in operators:
            # 查找没有适当空格的运算符
            pattern = rf'([^\\s])({re.escape(op)})([^\\s])'
            replacement = r'\1 \2 \3'
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text)
                self.error_stats['spacing_errors'] += 1
        
        return text
    
    def fix_escape_characters(self, text: str) -> str:
        """修复转义字符问题"""
        # 需要转义的字符
        escape_chars = ['{', '}', '[', ']', '^', '_', '\\', '&', '%', '$', '#']
        
        for char in escape_chars:
            # 查找未转义的特殊字符
            pattern = rf'(?<!\\){re.escape(char)}'
            if re.search(pattern, text):
                replacement = f'\\{char}'
                text = re.sub(pattern, replacement, text)
                self.error_stats['syntax_errors'] += 1
        
        return text
    
    def fix_greek_letters(self, text: str) -> str:
        """修复希腊字母使用"""
        # 常见错误映射
        greek_fixes = {
            'α': '\\alpha',
            'β': '\\beta',
            'γ': '\\gamma',
            'δ': '\\delta',
            'ε': '\\varepsilon',
            'ζ': '\\zeta',
            'η': '\\eta',
            'θ': '\\theta',
            'ι': '\\iota',
            'κ': '\\kappa',
            'λ': '\\lambda',
            'μ': '\\mu',
            'ν': '\\nu',
            'ξ': '\\xi',
            'ο': '\\omicron',
            'π': '\\pi',
            'ρ': '\\rho',
            'σ': '\\sigma',
            'τ': '\\tau',
            'υ': '\\upsilon',
            'φ': '\\phi',
            'χ': '\\chi',
            'ψ': '\\psi',
            'ω': '\\omega'
        }
        
        for unicode_char, latex_cmd in greek_fixes.items():
            if unicode_char in text:
                text = text.replace(unicode_char, f'${latex_cmd}$')
                self.error_stats['symbol_errors'] += 1
        
        return text
    
    def fix_math_symbols(self, text: str) -> str:
        """修复数学符号使用"""
        # 常见错误映射
        symbol_fixes = {
            '∈': '\\in',
            '∉': '\\notin',
            '⊂': '\\subset',
            '⊆': '\\subseteq',
            '⊃': '\\supset',
            '⊇': '\\supseteq',
            '∪': '\\cup',
            '∩': '\\cap',
            '∅': '\\emptyset',
            '∞': '\\infty',
            '±': '\\pm',
            '∓': '\\mp',
            '×': '\\times',
            '÷': '\\div',
            '·': '\\cdot',
            '≤': '\\leq',
            '≥': '\\geq',
            '≠': '\\neq',
            '≈': '\\approx',
            '≡': '\\equiv',
            '→': '\\rightarrow',
            '←': '\\leftarrow',
            '↔': '\\leftrightarrow',
            '⇒': '\\Rightarrow',
            '⇐': '\\Leftarrow',
            '⇔': '\\Leftrightarrow',
            '∀': '\\forall',
            '∃': '\\exists',
            '∄': '\\nexists',
            '∧': '\\land',
            '∨': '\\lor',
            '¬': '\\neg',
            '∴': '\\therefore',
            '∵': '\\because'
        }
        
        for unicode_char, latex_cmd in symbol_fixes.items():
            if unicode_char in text:
                text = text.replace(unicode_char, f'${latex_cmd}$')
                self.error_stats['symbol_errors'] += 1
        
        return text
    
    def fix_inline_formulas(self, text: str) -> str:
        """修复行内公式格式"""
        # 查找错误的块级公式在行内使用
        pattern = r'([^$])\$\$([^$]+)\$\$([^$])'
        
        def fix_inline(match):
            before = match.group(1)
            formula = match.group(2)
            after = match.group(3)
            self.error_stats['format_errors'] += 1
            return f'{before}${formula}${after}'
        
        text = re.sub(pattern, fix_inline, text)
        return text
    
    def fix_block_formulas(self, text: str) -> str:
        """修复块级公式格式"""
        # 查找长行内公式，应该改为块级
        inline_pattern = r'\$([^$]{50,})\$'
        
        def fix_block(match):
            formula = match.group(1)
            self.error_stats['format_errors'] += 1
            return f'\n$$\n{formula}\n$$\n'
        
        text = re.sub(inline_pattern, fix_block, text)
        return text
    
    def fix_alignment(self, text: str) -> str:
        """修复对齐问题"""
        # 修复align环境中的对齐
        pattern = r'\\begin\{align\}(.*?)\\end\{align\}'
        
        def fix_align(match):
            content = match.group(1)
            # 确保每行都有&符号
            lines = content.strip().split('\n')
            fixed_lines = []
            for line in lines:
                line = line.strip()
                if line and '&' not in line and '=' in line:
                    # 在等号前添加&
                    line = line.replace('=', '&=', 1)
                    self.error_stats['format_errors'] += 1
                fixed_lines.append(line)
            return '\\begin{align}\n' + '\n'.join(fixed_lines) + '\n\\end{align}'
        
        text = re.sub(pattern, fix_align, text, flags=re.DOTALL)
        return text
    
    def fix_file(self, file_path: Path) -> Dict[str, Any]:
        """修复单个文件"""
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 应用所有修复
            fixed_content = original_content
            fixed_content = self.fix_bracket_mismatch(fixed_content)
            fixed_content = self.fix_spacing(fixed_content)
            fixed_content = self.fix_escape_characters(fixed_content)
            fixed_content = self.fix_greek_letters(fixed_content)
            fixed_content = self.fix_math_symbols(fixed_content)
            fixed_content = self.fix_inline_formulas(fixed_content)
            fixed_content = self.fix_block_formulas(fixed_content)
            fixed_content = self.fix_alignment(fixed_content)
            
            # 如果内容有变化，写回文件
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                # 统计修复信息
                original_errors = self.count_errors(original_content)
                fixed_errors = self.count_errors(fixed_content)
                
                result = {
                    'file': str(file_path),
                    'status': 'fixed',
                    'original_errors': original_errors,
                    'fixed_errors': fixed_errors,
                    'improvement': original_errors - fixed_errors,
                    'error_types': self.error_stats.copy()
                }
            else:
                result = {
                    'file': str(file_path),
                    'status': 'no_changes',
                    'original_errors': 0,
                    'fixed_errors': 0,
                    'improvement': 0,
                    'error_types': self.error_stats.copy()
                }
            
            self.results['files_processed'] += 1
            self.results['fix_details'].append(result)
            
            return result
            
        except Exception as e:
            result = {
                'file': str(file_path),
                'status': 'error',
                'error': str(e),
                'original_errors': 0,
                'fixed_errors': 0,
                'improvement': 0,
                'error_types': self.error_stats.copy()
            }
            self.results['fix_details'].append(result)
            return result
    
    def count_errors(self, text: str) -> int:
        """统计文本中的错误数量"""
        error_count = 0
        
        # 检查括号不匹配
        inline_pattern = r'\$[^$]+\$'
        block_pattern = r'\$\$[^$]+\$\$'
        
        for pattern in [inline_pattern, block_pattern]:
            matches = re.findall(pattern, text)
            for match in matches:
                if match.count('(') != match.count(')'):
                    error_count += 1
        
        # 检查Unicode符号
        unicode_symbols = ['α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ω',
                          '∈', '∉', '⊂', '⊆', '∪', '∩', '∅', '∞', '±', '×', '÷', '≤', '≥', '≠', '≈', '≡']
        for symbol in unicode_symbols:
            error_count += text.count(symbol)
        
        # 检查格式错误
        if '$$' in text and not re.search(r'\$\$[^$]+\$\$', text):
            error_count += 1
        
        return error_count
    
    def fix_directory(self, directory_path: str = None) -> Dict[str, Any]:
        """修复目录中所有Markdown文件"""
        if directory_path is None:
            directory_path = self.project_path
        
        directory = Path(directory_path)
        md_files = list(directory.rglob('*.md'))
        
        print(f"找到 {len(md_files)} 个Markdown文件")
        
        for i, file_path in enumerate(md_files, 1):
            print(f"处理文件 {i}/{len(md_files)}: {file_path.name}")
            self.fix_file(file_path)
        
        # 生成总结报告
        self.generate_summary_report()
        
        return self.results
    
    def generate_summary_report(self):
        """生成总结报告"""
        total_files = len(self.results['fix_details'])
        fixed_files = len([r for r in self.results['fix_details'] if r['status'] == 'fixed'])
        error_files = len([r for r in self.results['fix_details'] if r['status'] == 'error'])
        
        total_improvement = sum([r.get('improvement', 0) for r in self.results['fix_details']])
        
        # 统计错误类型
        total_syntax_errors = sum([r.get('error_types', {}).get('syntax_errors', 0) for r in self.results['fix_details']])
        total_symbol_errors = sum([r.get('error_types', {}).get('symbol_errors', 0) for r in self.results['fix_details']])
        total_format_errors = sum([r.get('error_types', {}).get('format_errors', 0) for r in self.results['fix_details']])
        total_spacing_errors = sum([r.get('error_types', {}).get('spacing_errors', 0) for r in self.results['fix_details']])
        
        report = f"""
# 数学格式修复总结报告

## 总体统计
- 总文件数: {total_files}
- 成功修复: {fixed_files}
- 修复失败: {error_files}
- 总错误减少: {total_improvement}

## 错误类型统计
- 语法错误: {total_syntax_errors}
- 符号错误: {total_symbol_errors}
- 格式错误: {total_format_errors}
- 空格错误: {total_spacing_errors}

## 详细结果
"""
        
        for result in self.results['fix_details']:
            if result['status'] == 'error':
                report += f"- {result['file']}: 错误 - {result['error']}\n"
            else:
                report += f"- {result['file']}: 修复 {result['improvement']} 个错误\n"
        
        # 保存报告
        report_path = self.project_path / '数学格式修复报告.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"修复报告已保存到: {report_path}")
        print(report)

class MathFormatChecker:
    """数学格式检查器"""
    
    def __init__(self, project_path: str):
        """初始化检查器"""
        self.project_path = Path(project_path)
        self.check_results = []
    
    def check_math_syntax(self, text: str) -> List[str]:
        """检查数学公式语法"""
        errors = []
        
        # 检查行内公式
        inline_pattern = r'\$[^$]+\$'
        inline_matches = re.findall(inline_pattern, text)
        
        for match in inline_matches:
            # 检查括号匹配
            if match.count('(') != match.count(')'):
                errors.append(f"括号不匹配: {match}")
            
            # 检查反斜杠转义
            if '\\' in match and not re.search(r'\\[a-zA-Z]', match):
                errors.append(f"反斜杠转义错误: {match}")
            
            # 检查是否应该是块级公式
            if len(match) > 50:
                errors.append(f"长公式应使用块级格式: {match}")
        
        # 检查块级公式
        block_pattern = r'\$\$[^$]+\$\$'
        block_matches = re.findall(block_pattern, text)
        
        for match in block_matches:
            # 检查换行
            if '\n' in match:
                errors.append(f"块级公式包含换行: {match}")
            
            # 检查是否应该是行内公式
            if len(match) < 20:
                errors.append(f"短公式应使用行内格式: {match}")
        
        return errors
    
    def check_math_symbols(self, text: str) -> List[str]:
        """检查数学符号使用"""
        errors = []
        
        # 检查Unicode符号使用
        unicode_symbols = ['α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ω',
                          '∈', '∉', '⊂', '⊆', '∪', '∩', '∅', '∞', '±', '×', '÷', '≤', '≥', '≠', '≈', '≡']
        
        for symbol in unicode_symbols:
            if symbol in text:
                errors.append(f"应使用LaTeX命令而不是Unicode符号: {symbol}")
        
        return errors
    
    def check_math_format(self, text: str) -> List[str]:
        """检查数学格式"""
        errors = []
        
        # 检查行内公式格式
        if '$$' in text and not re.search(r'\$\$[^$]+\$\$', text):
            errors.append("行内公式不应使用$$")
        
        # 检查块级公式格式
        if '$' in text and re.search(r'\$[^$]{50,}\$', text):
            errors.append("长公式应使用块级格式")
        
        # 检查对齐格式
        align_pattern = r'\\begin\{align\}(.*?)\\end\{align\}'
        align_matches = re.findall(align_pattern, text, flags=re.DOTALL)
        
        for match in align_matches:
            lines = match.strip().split('\n')
            for line in lines:
                if line.strip() and '=' in line and '&' not in line:
                    errors.append(f"align环境中的等式应使用&对齐: {line}")
        
        return errors
    
    def check_file(self, file_path: Path) -> Dict[str, Any]:
        """检查单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            syntax_errors = self.check_math_syntax(content)
            symbol_errors = self.check_math_symbols(content)
            format_errors = self.check_math_format(content)
            
            result = {
                'file': str(file_path),
                'syntax_errors': syntax_errors,
                'symbol_errors': symbol_errors,
                'format_errors': format_errors,
                'total_errors': len(syntax_errors) + len(symbol_errors) + len(format_errors)
            }
            
            self.check_results.append(result)
            return result
            
        except Exception as e:
            result = {
                'file': str(file_path),
                'error': str(e),
                'syntax_errors': [],
                'symbol_errors': [],
                'format_errors': [],
                'total_errors': 0
            }
            self.check_results.append(result)
            return result
    
    def check_directory(self, directory_path: str = None) -> Dict[str, Any]:
        """检查目录中所有Markdown文件"""
        if directory_path is None:
            directory_path = self.project_path
        
        directory = Path(directory_path)
        md_files = list(directory.rglob('*.md'))
        
        print(f"检查 {len(md_files)} 个Markdown文件")
        
        for i, file_path in enumerate(md_files, 1):
            print(f"检查文件 {i}/{len(md_files)}: {file_path.name}")
            self.check_file(file_path)
        
        # 生成检查报告
        self.generate_check_report()
        
        return {
            'total_files': len(md_files),
            'check_results': self.check_results
        }
    
    def generate_check_report(self):
        """生成检查报告"""
        total_files = len(self.check_results)
        total_syntax_errors = sum(len(r.get('syntax_errors', [])) for r in self.check_results)
        total_symbol_errors = sum(len(r.get('symbol_errors', [])) for r in self.check_results)
        total_format_errors = sum(len(r.get('format_errors', [])) for r in self.check_results)
        
        report = f"""
# 数学格式检查报告

## 总体统计
- 检查文件数: {total_files}
- 语法错误: {total_syntax_errors}
- 符号错误: {total_symbol_errors}
- 格式错误: {total_format_errors}
- 总错误数: {total_syntax_errors + total_symbol_errors + total_format_errors}

## 详细错误
"""
        
        for result in self.check_results:
            if 'error' in result:
                report += f"\n### {result['file']}\n- 文件读取错误: {result['error']}\n"
            else:
                report += f"\n### {result['file']}\n"
                if result['syntax_errors']:
                    report += "- 语法错误:\n"
                    for err in result['syntax_errors']:
                        report += f"  - {err}\n"
                if result['symbol_errors']:
                    report += "- 符号错误:\n"
                    for err in result['symbol_errors']:
                        report += f"  - {err}\n"
                if result['format_errors']:
                    report += "- 格式错误:\n"
                    for err in result['format_errors']:
                        report += f"  - {err}\n"
        
        # 保存报告
        report_path = self.project_path / '数学格式检查报告.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"检查报告已保存到: {report_path}")
        print(report)

def main():
    """主函数"""
    print("数学格式修复工具")
    print("=" * 50)
    
    # 获取项目路径
    project_path = input("请输入项目路径 (默认为当前目录): ").strip()
    if not project_path:
        project_path = "."
    
    # 选择操作模式
    print("\n请选择操作模式:")
    print("1. 检查数学格式错误")
    print("2. 修复数学格式错误")
    print("3. 检查并修复")
    
    mode = input("请输入选择 (1/2/3): ").strip()
    
    if mode == "1":
        # 检查模式
        checker = MathFormatChecker(project_path)
        checker.check_directory()
        
    elif mode == "2":
        # 修复模式
        fixer = MathFormatFixer(project_path)
        fixer.fix_directory()
        
    elif mode == "3":
        # 检查并修复模式
        print("\n第一步: 检查数学格式错误")
        checker = MathFormatChecker(project_path)
        checker.check_directory()
        
        print("\n第二步: 修复数学格式错误")
        fixer = MathFormatFixer(project_path)
        fixer.fix_directory()
        
    else:
        print("无效的选择")
        return
    
    print("\n操作完成!")

if __name__ == "__main__":
    main() 