#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复测试套件
提供全面的单元测试、集成测试和性能测试
"""

import unittest
import json
import time
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

# 模拟核心模块（实际使用时需要导入真实模块）
class MockMathFormatFixer:
    """模拟数学格式修复器"""
    
    def __init__(self):
        self.config = {}
    
    def update_config(self, config: Dict[str, Any]):
        self.config = config
    
    def fix_text(self, text: str) -> str:
        """模拟修复文本"""
        # 简单的修复规则
        fixed = text
        fixed = fixed.replace('$', '$$')  # 单美元符号转双美元符号
        fixed = fixed.replace('\\[', '$$')  # 行间公式
        fixed = fixed.replace('\\]', '$$')
        fixed = fixed.replace('\\(', '$')   # 行内公式
        fixed = fixed.replace('\\)', '$')
        return fixed
    
    def check_format(self, text: str) -> List[Dict[str, Any]]:
        """模拟检查格式"""
        issues = []
        if '$' in text and '$$' not in text:
            issues.append({
                'type': 'single_dollar',
                'message': '发现单美元符号，建议使用双美元符号',
                'line': 1,
                'column': text.find('$') + 1
            })
        return issues

class MockConfigManager:
    """模拟配置管理器"""
    
    def __init__(self):
        self.config = {}
    
    def load_config(self, config_path: str):
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            raise ValueError(f"配置文件加载失败: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return self.config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        required_keys = ['rules', 'settings']
        return all(key in config for key in required_keys)

class TestMathFormatFixer(unittest.TestCase):
    """数学格式修复器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.fixer = MockMathFormatFixer()
        self.config_manager = MockConfigManager()
        
        # 测试配置
        self.test_config = {
            'rules': {
                'dollar_signs': True,
                'brackets': True,
                'spacing': True
            },
            'settings': {
                'output_format': 'markdown',
                'preserve_original': True
            }
        }
    
    def test_fix_text_basic(self):
        """测试基本文本修复"""
        test_text = "这是一个公式: $x^2 + y^2 = z^2$"
        expected = "这是一个公式: $$x^2 + y^2 = z^2$$"
        result = self.fixer.fix_text(test_text)
        self.assertEqual(result, expected)
    
    def test_fix_text_brackets(self):
        """测试括号修复"""
        test_text = "行间公式: \\[E = mc^2\\] 和行内公式: \\(a + b\\)"
        expected = "行间公式: $$E = mc^2$$ 和行内公式: $a + b$"
        result = self.fixer.fix_text(test_text)
        self.assertEqual(result, expected)
    
    def test_check_format_issues(self):
        """测试格式检查"""
        test_text = "有问题的公式: $x^2$ 和正确的公式: $$y^2$$"
        issues = self.fixer.check_format(test_text)
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0]['type'], 'single_dollar')
    
    def test_check_format_no_issues(self):
        """测试无问题的格式检查"""
        test_text = "正确的公式: $$x^2 + y^2 = z^2$$"
        issues = self.fixer.check_format(test_text)
        self.assertEqual(len(issues), 0)

class TestConfigManager(unittest.TestCase):
    """配置管理器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config_manager = MockConfigManager()
        
        # 创建临时配置文件
        self.temp_config = {
            'rules': {
                'dollar_signs': True,
                'brackets': True
            },
            'settings': {
                'output_format': 'markdown'
            }
        }
    
    def test_load_config_valid(self):
        """测试加载有效配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.temp_config, f)
            config_path = f.name
        
        try:
            self.config_manager.load_config(config_path)
            config = self.config_manager.get_config()
            self.assertEqual(config, self.temp_config)
        finally:
            os.unlink(config_path)
    
    def test_load_config_invalid(self):
        """测试加载无效配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.config_manager.load_config(config_path)
        finally:
            os.unlink(config_path)
    
    def test_validate_config_valid(self):
        """测试验证有效配置"""
        self.assertTrue(self.config_manager.validate_config(self.temp_config))
    
    def test_validate_config_invalid(self):
        """测试验证无效配置"""
        invalid_config = {'rules': {}}  # 缺少settings
        self.assertFalse(self.config_manager.validate_config(invalid_config))

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.fixer = MockMathFormatFixer()
        self.config_manager = MockConfigManager()
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 1. 加载配置
        config = {
            'rules': {'dollar_signs': True},
            'settings': {'output_format': 'markdown'}
        }
        self.config_manager.config = config
        self.fixer.update_config(config)
        
        # 2. 处理文本
        test_text = "公式: $x^2$ 和 $y^2$"
        fixed_text = self.fixer.fix_text(test_text)
        
        # 3. 检查结果
        self.assertIn('$$', fixed_text)
        self.assertNotIn('$', fixed_text)
    
    def test_batch_processing(self):
        """测试批量处理"""
        test_files = [
            "文件1: $a + b$",
            "文件2: $$c + d$$",
            "文件3: $e + f$"
        ]
        
        results = []
        for i, content in enumerate(test_files):
            fixed = self.fixer.fix_text(content)
            issues = self.fixer.check_format(content)
            results.append({
                'file_id': i,
                'original': content,
                'fixed': fixed,
                'issues': len(issues)
            })
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r['issues'] > 0 for r in results if '$' in r['original'] and '$$' not in r['original']))

class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.fixer = MockMathFormatFixer()
    
    def test_large_text_processing(self):
        """测试大文本处理性能"""
        # 生成大文本
        large_text = "测试文本 " * 1000 + "$x^2$ " * 100
        
        start_time = time.time()
        fixed_text = self.fixer.fix_text(large_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0)  # 应该在1秒内完成
        self.assertIn('$$', fixed_text)
    
    def test_multiple_files_performance(self):
        """测试多文件处理性能"""
        files = [f"文件{i}: $x^{i}$" for i in range(100)]
        
        start_time = time.time()
        results = []
        for content in files:
            fixed = self.fixer.fix_text(content)
            issues = self.fixer.check_format(content)
            results.append({
                'fixed': fixed,
                'issues': len(issues)
            })
        end_time = time.time()
        
        processing_time = end_time - start_time
        self.assertLess(processing_time, 2.0)  # 应该在2秒内完成
        self.assertEqual(len(results), 100)

class TestErrorHandling(unittest.TestCase):
    """错误处理测试"""
    
    def setUp(self):
        """测试前准备"""
        self.fixer = MockMathFormatFixer()
    
    def test_empty_text(self):
        """测试空文本处理"""
        result = self.fixer.fix_text("")
        self.assertEqual(result, "")
        
        issues = self.fixer.check_format("")
        self.assertEqual(len(issues), 0)
    
    def test_none_text(self):
        """测试None文本处理"""
        with self.assertRaises(AttributeError):
            self.fixer.fix_text(None)
    
    def test_special_characters(self):
        """测试特殊字符处理"""
        test_text = "特殊字符: $@#$%^&*()$ 和 $$数学公式$$"
        result = self.fixer.fix_text(test_text)
        self.assertIn('$$', result)
    
    def test_unicode_characters(self):
        """测试Unicode字符处理"""
        test_text = "中文公式: $α + β = γ$ 和 $$∑_{i=1}^n x_i$$"
        result = self.fixer.fix_text(test_text)
        self.assertIn('$$', result)
        self.assertIn('α', result)
        self.assertIn('β', result)

class TestRegression(unittest.TestCase):
    """回归测试"""
    
    def setUp(self):
        """测试前准备"""
        self.fixer = MockMathFormatFixer()
    
    def test_regression_known_cases(self):
        """测试已知案例的回归"""
        test_cases = [
            ("$x$", "$$x$$"),
            ("$$y$$", "$$y$$"),
            ("\\(z\\)", "$z$"),
            ("\\[w\\]", "$$w$$"),
            ("混合: $a$ 和 $$b$$", "混合: $$a$$ 和 $$b$$")
        ]
        
        for original, expected in test_cases:
            with self.subTest(original=original):
                result = self.fixer.fix_text(original)
                self.assertEqual(result, expected)

def run_tests():
    """运行所有测试"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestMathFormatFixer,
        TestConfigManager,
        TestIntegration,
        TestPerformance,
        TestErrorHandling,
        TestRegression
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果统计
    print(f"\n测试结果统计:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 