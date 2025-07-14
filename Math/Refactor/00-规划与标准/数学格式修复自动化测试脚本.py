#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复自动化测试脚本
Math/Refactor项目数学格式修复工具自动化测试

作者: 数学知识体系重构项目组
时间: 2025年1月
版本: 1.0
"""

import os
import re
import time
import json
import unittest
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# 导入修复工具
import sys
sys.path.append(str(Path(__file__).parent))
from 数学格式修复执行脚本 import MathFormatFixer, MathFormatChecker

class MathFormatTestSuite(unittest.TestCase):
    """数学格式修复测试套件"""
    
    def setUp(self):
        """测试前准备"""
        self.fixer = MathFormatFixer("test")
        self.checker = MathFormatChecker("test")
        self.test_results = []
        self.performance_data = {
            'avg_time': 0,
            'max_time': 0,
            'min_time': float('inf'),
            'peak_memory': 0,
            'syntax_performance': 'good',
            'symbol_performance': 'good',
            'format_performance': 'good'
        }
    
    def tearDown(self):
        """测试后清理"""
        pass
    
    def record_test_result(self, test_name: str, status: str, description: str, error: str = None):
        """记录测试结果"""
        result = {
            'name': test_name,
            'status': status,
            'description': description,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
    
    def measure_performance(self, func, *args, **kwargs):
        """测量性能"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 更新性能数据
        self.performance_data['avg_time'] = (self.performance_data['avg_time'] + processing_time) / 2
        self.performance_data['max_time'] = max(self.performance_data['max_time'], processing_time)
        self.performance_data['min_time'] = min(self.performance_data['min_time'], processing_time)
        
        return result, processing_time

class SyntaxErrorTests(MathFormatTestSuite):
    """语法错误测试类"""
    
    def test_bracket_mismatch_fix(self):
        """测试括号不匹配修复"""
        test_cases = [
            ("$f(x = x^2 + 1$", "$f(x) = x^2 + 1$", "缺失右括号"),
            ("$f(x) = x^2 + 1)$", "$f(x) = x^2 + 1$", "缺失左括号"),
            ("$f(x = (x + 1)(x - 1$", "$f(x) = (x + 1)(x - 1)$", "多重括号不匹配"),
            ("$f(x = \\frac{x + 1}{x - 1}$", "$f(x) = \\frac{x + 1}{x - 1}$", "嵌套括号不匹配")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_bracket_mismatch, input_text
                )
                self.assertEqual(result, expected, f"括号修复失败: {description}")
                self.record_test_result(
                    f"bracket_fix_{description}",
                    "passed",
                    f"括号修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"bracket_fix_{description}",
                    "failed",
                    f"括号修复测试: {description}",
                    str(e)
                )
    
    def test_escape_character_fix(self):
        """测试转义字符修复"""
        test_cases = [
            ("$f(x) = {x^2 + 1}$", "$f(x) = \\{x^2 + 1\\}$", "未转义的大括号"),
            ("$f(x) = x^2 + 1 & x > 0$", "$f(x) = x^2 + 1 \\& x > 0$", "未转义的&符号"),
            ("$f(x) = 50% \\times x$", "$f(x) = 50\\% \\times x$", "未转义的百分号"),
            ("$f(x) = $x^2 + 1$", "$f(x) = \\$x^2 + 1$", "未转义的美元符号")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_escape_characters, input_text
                )
                self.assertEqual(result, expected, f"转义字符修复失败: {description}")
                self.record_test_result(
                    f"escape_fix_{description}",
                    "passed",
                    f"转义字符修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"escape_fix_{description}",
                    "failed",
                    f"转义字符修复测试: {description}",
                    str(e)
                )
    
    def test_spacing_fix(self):
        """测试空格修复"""
        test_cases = [
            ("$f(x)=x^2+1$", "$f(x) = x^2 + 1$", "运算符前后缺少空格"),
            ("$x<y$ 和 $a>=b$", "$x < y$ 和 $a >= b$", "比较运算符缺少空格"),
            ("$x\\land y$ 和 $a\\lor b$", "$x \\land y$ 和 $a \\lor b$", "逻辑运算符缺少空格")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_spacing, input_text
                )
                self.assertEqual(result, expected, f"空格修复失败: {description}")
                self.record_test_result(
                    f"spacing_fix_{description}",
                    "passed",
                    f"空格修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"spacing_fix_{description}",
                    "failed",
                    f"空格修复测试: {description}",
                    str(e)
                )

class SymbolErrorTests(MathFormatTestSuite):
    """符号错误测试类"""
    
    def test_greek_letter_fix(self):
        """测试希腊字母修复"""
        test_cases = [
            ("$α + β = γ$", "$\\alpha + \\beta = \\gamma$", "Unicode希腊字母"),
            ("$Α + Β = Γ$", "$\\Alpha + \\Beta = \\Gamma$", "大写希腊字母"),
            ("$αβγδεζηθικλμνξοπρστυφχψω$", "$\\alpha\\beta\\gamma\\delta\\varepsilon\\zeta\\eta\\theta\\iota\\kappa\\lambda\\mu\\nu\\xi\\omicron\\pi\\rho\\sigma\\tau\\upsilon\\phi\\chi\\psi\\omega$", "混合希腊字母")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_greek_letters, input_text
                )
                self.assertEqual(result, expected, f"希腊字母修复失败: {description}")
                self.record_test_result(
                    f"greek_fix_{description}",
                    "passed",
                    f"希腊字母修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"greek_fix_{description}",
                    "failed",
                    f"希腊字母修复测试: {description}",
                    str(e)
                )
    
    def test_math_symbol_fix(self):
        """测试数学符号修复"""
        test_cases = [
            ("$x ∈ A$ 和 $A ⊂ B$", "$x \\in A$ 和 $A \\subset B$", "集合符号"),
            ("$x ≤ y$ 和 $a ≥ b$ 和 $c ≠ d$", "$x \\leq y$ 和 $a \\geq b$ 和 $c \\neq d$", "比较符号"),
            ("$a × b$ 和 $c ÷ d$ 和 $e ± f$", "$a \\times b$ 和 $c \\div d$ 和 $e \\pm f$", "运算符符号"),
            ("$∀x ∃y$ 和 $p ∧ q$ 和 $r ∨ s$", "$\\forall x \\exists y$ 和 $p \\land q$ 和 $r \\lor s$", "逻辑符号")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_math_symbols, input_text
                )
                self.assertEqual(result, expected, f"数学符号修复失败: {description}")
                self.record_test_result(
                    f"symbol_fix_{description}",
                    "passed",
                    f"数学符号修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"symbol_fix_{description}",
                    "failed",
                    f"数学符号修复测试: {description}",
                    str(e)
                )
    
    def test_calculus_symbol_fix(self):
        """测试微积分符号修复"""
        test_cases = [
            ("$∫f(x)dx$ 和 $∮f(z)dz$", "$\\int f(x)dx$ 和 $\\oint f(z)dz$", "积分符号"),
            ("$∑_{i=1}^n x_i$ 和 $∏_{i=1}^n x_i$", "$\\sum_{i=1}^n x_i$ 和 $\\prod_{i=1}^n x_i$", "求和符号"),
            ("$lim_{x→∞} f(x)$", "$\\lim_{x \\to \\infty} f(x)$", "极限符号")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_math_symbols, input_text
                )
                self.assertEqual(result, expected, f"微积分符号修复失败: {description}")
                self.record_test_result(
                    f"calculus_fix_{description}",
                    "passed",
                    f"微积分符号修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"calculus_fix_{description}",
                    "failed",
                    f"微积分符号修复测试: {description}",
                    str(e)
                )

class FormatErrorTests(MathFormatTestSuite):
    """格式错误测试类"""
    
    def test_inline_formula_fix(self):
        """测试行内公式修复"""
        test_cases = [
            ("这是一个公式：$$f(x) = x^2$$", "这是一个公式：$f(x) = x^2$", "行内使用块级格式"),
            ("$f(x) = \\int_{-\\infty}^{\\infty} \\hat{f}(\\xi)\\,e^{2 \\pi i \\xi x} \\,d\\xi$", "\n$$\nf(x) = \\int_{-\\infty}^{\\infty} \\hat{f}(\\xi)\\,e^{2 \\pi i \\xi x} \\,d\\xi\n$$\n", "长行内公式"),
            ("$f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$", "\n$$\nf(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}\n$$\n", "复杂行内公式")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                if "长行内公式" in description or "复杂行内公式" in description:
                    result, processing_time = self.measure_performance(
                        self.fixer.fix_block_formulas, input_text
                    )
                else:
                    result, processing_time = self.measure_performance(
                        self.fixer.fix_inline_formulas, input_text
                    )
                self.assertEqual(result, expected, f"行内公式修复失败: {description}")
                self.record_test_result(
                    f"inline_fix_{description}",
                    "passed",
                    f"行内公式修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"inline_fix_{description}",
                    "failed",
                    f"行内公式修复测试: {description}",
                    str(e)
                )
    
    def test_block_formula_fix(self):
        """测试块级公式修复"""
        test_cases = [
            ("$$E = mc^2$$", "$E = mc^2$", "短块级公式"),
            ("$$\nf(x) = x^2 + 2x + 1\n$$", "$$\nf(x) = x^2 + 2x + 1\n$$", "块级公式换行错误")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_block_formulas, input_text
                )
                self.assertEqual(result, expected, f"块级公式修复失败: {description}")
                self.record_test_result(
                    f"block_fix_{description}",
                    "passed",
                    f"块级公式修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"block_fix_{description}",
                    "failed",
                    f"块级公式修复测试: {description}",
                    str(e)
                )
    
    def test_matrix_format_fix(self):
        """测试矩阵格式修复"""
        test_cases = [
            ("$$\n\\begin{pmatrix}\na & b\nc & d\n\\end{pmatrix}\n$$", "$$\n\\begin{pmatrix}\na & b \\\\\nc & d\n\\end{pmatrix}\n$$", "矩阵缺少换行"),
            ("$$\n\\begin{bmatrix}\na & b & c\nd & e & f\ng & h & i\n\\end{bmatrix}\n$$", "$$\n\\begin{bmatrix}\na & b & c \\\\\nd & e & f \\\\\ng & h & i\n\\end{bmatrix}\n$$", "矩阵对齐错误")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_alignment, input_text
                )
                self.assertEqual(result, expected, f"矩阵格式修复失败: {description}")
                self.record_test_result(
                    f"matrix_fix_{description}",
                    "passed",
                    f"矩阵格式修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"matrix_fix_{description}",
                    "failed",
                    f"矩阵格式修复测试: {description}",
                    str(e)
                )
    
    def test_equation_system_fix(self):
        """测试方程组格式修复"""
        test_cases = [
            ("$$\n\\begin{align}\nx + y = 1\n2x - y = 0\n\\end{align}\n$$", "$$\n\\begin{align}\nx + y &= 1 \\\\\n2x - y &= 0\n\\end{align}\n$$", "align环境缺少对齐符号"),
            ("$$\n\\begin{cases}\nx^2 & \\text{if } x > 0\n0 & \\text{if } x = 0\n-x^2 & \\text{if } x < 0\n\\end{cases}\n$$", "$$\n\\begin{cases}\nx^2 & \\text{if } x > 0 \\\\\n0 & \\text{if } x = 0 \\\\\n-x^2 & \\text{if } x < 0\n\\end{cases}\n$$", "cases环境格式错误")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                result, processing_time = self.measure_performance(
                    self.fixer.fix_alignment, input_text
                )
                self.assertEqual(result, expected, f"方程组格式修复失败: {description}")
                self.record_test_result(
                    f"equation_fix_{description}",
                    "passed",
                    f"方程组格式修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"equation_fix_{description}",
                    "failed",
                    f"方程组格式修复测试: {description}",
                    str(e)
                )

class ComplexScenarioTests(MathFormatTestSuite):
    """复杂场景测试类"""
    
    def test_mixed_error_fix(self):
        """测试混合错误修复"""
        test_cases = [
            ("$f(x = αx^2 + βx + γ$ 和 $g(x) = ∫f(t)dt$", "$f(x) = \\alpha x^2 + \\beta x + \\gamma$ 和 $g(x) = \\int f(t)dt$", "多种错误混合"),
            ("$$\nf(x) = \\begin{cases}\nx^2 & \\text{if } x > 0\n0 & \\text{if } x = 0\n-x^2 & \\text{if } x < 0\n\\end{cases}\n$$", "$$\nf(x) = \\begin{cases}\nx^2 & \\text{if } x > 0 \\\\\n0 & \\text{if } x = 0 \\\\\n-x^2 & \\text{if } x < 0\n\\end{cases}\n$$", "复杂公式混合错误")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                # 应用所有修复
                result = input_text
                result, _ = self.measure_performance(self.fixer.fix_bracket_mismatch, result)
                result, _ = self.measure_performance(self.fixer.fix_greek_letters, result)
                result, _ = self.measure_performance(self.fixer.fix_math_symbols, result)
                result, _ = self.measure_performance(self.fixer.fix_inline_formulas, result)
                result, _ = self.measure_performance(self.fixer.fix_block_formulas, result)
                result, processing_time = self.measure_performance(self.fixer.fix_alignment, result)
                
                self.assertEqual(result, expected, f"混合错误修复失败: {description}")
                self.record_test_result(
                    f"mixed_fix_{description}",
                    "passed",
                    f"混合错误修复测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"mixed_fix_{description}",
                    "failed",
                    f"混合错误修复测试: {description}",
                    str(e)
                )
    
    def test_boundary_cases(self):
        """测试边界情况"""
        test_cases = [
            ("$$ $$", "$$ $$", "空公式"),
            ("$x$", "$x$", "单个字符公式"),
            ("$f(x) = \\int_{-\\infty}^{\\infty} \\hat{f}(\\xi)\\,e^{2 \\pi i \\xi x} \\,d\\xi + \\sum_{n=1}^{\\infty} \\frac{a_n}{n!}x^n + \\prod_{k=1}^{n} (1 + \\frac{x}{k})$", "\n$$\nf(x) = \\int_{-\\infty}^{\\infty} \\hat{f}(\\xi)\\,e^{2 \\pi i \\xi x} \\,d\\xi + \\sum_{n=1}^{\\infty} \\frac{a_n}{n!}x^n + \\prod_{k=1}^{n} (1 + \\frac{x}{k})\n$$\n", "非常长的公式")
        ]
        
        for input_text, expected, description in test_cases:
            try:
                if "非常长的公式" in description:
                    result, processing_time = self.measure_performance(
                        self.fixer.fix_block_formulas, input_text
                    )
                else:
                    result, processing_time = self.measure_performance(
                        lambda x: x, input_text
                    )
                self.assertEqual(result, expected, f"边界情况处理失败: {description}")
                self.record_test_result(
                    f"boundary_{description}",
                    "passed",
                    f"边界情况测试: {description}",
                )
            except Exception as e:
                self.record_test_result(
                    f"boundary_{description}",
                    "failed",
                    f"边界情况测试: {description}",
                    str(e)
                )

class PerformanceTests(MathFormatTestSuite):
    """性能测试类"""
    
    def test_large_file_performance(self):
        """测试大文件性能"""
        try:
            # 创建大文件内容
            large_content = "$f(x) = x^2$" * 1000
            
            start_time = time.time()
            result = large_content
            result, _ = self.measure_performance(self.fixer.fix_bracket_mismatch, result)
            result, _ = self.measure_performance(self.fixer.fix_greek_letters, result)
            result, _ = self.measure_performance(self.fixer.fix_math_symbols, result)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # 性能要求：处理1000个公式不超过5秒
            self.assertLess(processing_time, 5.0, f"性能回归: {processing_time}秒")
            self.record_test_result(
                "large_file_performance",
                "passed",
                f"大文件性能测试: 处理1000个公式用时{processing_time:.3f}秒",
            )
        except Exception as e:
            self.record_test_result(
                "large_file_performance",
                "failed",
                "大文件性能测试",
                str(e)
            )
    
    def test_memory_usage(self):
        """测试内存使用"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # 执行大量操作
            for i in range(100):
                content = f"$f(x) = x^{i}$" * 100
                self.fixer.fix_bracket_mismatch(content)
                self.fixer.fix_greek_letters(content)
                self.fixer.fix_math_symbols(content)
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # 内存增长不应超过100MB
            self.assertLess(memory_increase, 100, f"内存使用过高: 增长{memory_increase:.2f}MB")
            self.record_test_result(
                "memory_usage",
                "passed",
                f"内存使用测试: 增长{memory_increase:.2f}MB",
            )
        except Exception as e:
            self.record_test_result(
                "memory_usage",
                "failed",
                "内存使用测试",
                str(e)
            )

class IntegrationTests(MathFormatTestSuite):
    """集成测试类"""
    
    def test_complete_fix_workflow(self):
        """测试完整修复工作流程"""
        try:
            # 创建复杂的测试文档
            complex_document = """
            这是一个复杂的数学文档：
            
            $f(x = αx^2 + βx + γ$ 和 $g(x) = ∫f(t)dt$
            
            $$
            \begin{cases}
            x^2 & \text{if } x > 0
            0 & \text{if } x = 0
            -x^2 & \text{if } x < 0
            \end{cases}
            $$
            
            还有更多的公式：$∀x ∃y (x + y = 0)$
            """
            
            # 应用所有修复
            result, processing_time = self.measure_performance(
                lambda: self.fixer.fix_directory("test_directory"), 
                None
            )
            
            # 验证结果包含预期的修复
            self.assertIn("f(x) = \\alpha x^2 + \\beta x + \\gamma", str(result))
            self.assertIn("g(x) = \\int f(t)dt", str(result))
            self.assertIn("\\begin{cases}", str(result))
            self.assertIn("\\\\", str(result))  # 检查换行符
            
            self.record_test_result(
                "complete_workflow",
                "passed",
                f"完整工作流程测试: 用时{processing_time:.3f}秒",
            )
        except Exception as e:
            self.record_test_result(
                "complete_workflow",
                "failed",
                "完整工作流程测试",
                str(e)
            )
    
    def test_check_and_fix_workflow(self):
        """测试检查和修复工作流程"""
        try:
            # 创建测试文件
            test_content = "$f(x = αx^2 + βx + γ$"
            
            # 先检查
            check_results = self.checker.check_math_syntax(test_content)
            self.assertGreater(len(check_results), 0, "应该检测到语法错误")
            
            # 再修复
            fixed_content = test_content
            fixed_content = self.fixer.fix_bracket_mismatch(fixed_content)
            fixed_content = self.fixer.fix_greek_letters(fixed_content)
            
            # 再次检查
            check_results_after = self.checker.check_math_syntax(fixed_content)
            self.assertEqual(len(check_results_after), 0, "修复后应该没有语法错误")
            
            self.record_test_result(
                "check_and_fix_workflow",
                "passed",
                "检查和修复工作流程测试",
            )
        except Exception as e:
            self.record_test_result(
                "check_and_fix_workflow",
                "failed",
                "检查和修复工作流程测试",
                str(e)
            )

def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        SyntaxErrorTests,
        SymbolErrorTests,
        FormatErrorTests,
        ComplexScenarioTests,
        PerformanceTests,
        IntegrationTests
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

def generate_test_report(test_results, performance_data):
    """生成测试报告"""
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r['status'] == 'passed')
    failed_tests = sum(1 for r in test_results if r['status'] == 'failed')
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    report = f"""
# 数学格式修复自动化测试报告

## 测试概览
- 总测试用例数: {total_tests}
- 通过测试数: {passed_tests}
- 失败测试数: {failed_tests}
- 通过率: {pass_rate:.2f}%

## 性能指标
- 平均处理时间: {performance_data['avg_time']:.3f}秒
- 最大处理时间: {performance_data['max_time']:.3f}秒
- 最小处理时间: {performance_data['min_time']:.3f}秒
- 内存使用峰值: {performance_data['peak_memory']:.2f}MB

## 详细结果
"""
    
    # 按状态分组
    passed_results = [r for r in test_results if r['status'] == 'passed']
    failed_results = [r for r in test_results if r['status'] == 'failed']
    
    if passed_results:
        report += "\n### ✅ 通过的测试\n"
        for result in passed_results:
            report += f"- {result['name']}: {result['description']}\n"
    
    if failed_results:
        report += "\n### ❌ 失败的测试\n"
        for result in failed_results:
            report += f"- {result['name']}: {result['description']}\n"
            report += f"  错误: {result['error']}\n"
    
    # 性能分析
    report += f"""
## 性能分析
- 语法修复性能: {performance_data['syntax_performance']}
- 符号修复性能: {performance_data['symbol_performance']}
- 格式修复性能: {performance_data['format_performance']}

## 测试结论
"""
    
    if pass_rate >= 98:
        report += "✅ 测试通过率优秀，工具质量良好"
    elif pass_rate >= 95:
        report += "⚠️ 测试通过率良好，需要关注失败用例"
    else:
        report += "❌ 测试通过率较低，需要修复问题"
    
    return report

def main():
    """主函数"""
    print("数学格式修复自动化测试")
    print("=" * 50)
    
    # 运行测试
    print("开始运行测试...")
    result = run_all_tests()
    
    # 生成报告
    print("生成测试报告...")
    report = generate_test_report([], {})  # 这里需要实际的测试结果
    
    # 保存报告
    report_path = Path("数学格式修复测试报告.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"测试报告已保存到: {report_path}")
    print("测试完成!")

if __name__ == "__main__":
    main() 