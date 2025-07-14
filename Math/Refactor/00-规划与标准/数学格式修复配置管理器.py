#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复配置管理器
Math/Refactor项目数学格式修复配置管理工具

作者: 数学知识体系重构项目组
时间: 2025年1月
版本: 1.0
"""

import os
import json
import yaml
import configparser
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class FixRule:
    """修复规则配置"""
    name: str
    description: str
    enabled: bool = True
    priority: int = 0
    pattern: str = ""
    replacement: str = ""
    conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}

@dataclass
class CheckRule:
    """检查规则配置"""
    name: str
    description: str
    enabled: bool = True
    severity: str = "warning"  # info, warning, error
    pattern: str = ""
    message: str = ""
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

@dataclass
class ProcessingConfig:
    """处理配置"""
    max_workers: int = 4
    backup_enabled: bool = True
    recursive_processing: bool = True
    file_patterns: List[str] = None
    exclude_patterns: List[str] = None
    log_level: str = "INFO"
    output_format: str = "json"  # json, yaml, text
    
    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = ["*.md"]
        if self.exclude_patterns is None:
            self.exclude_patterns = ["*.bak", "*.tmp"]

@dataclass
class QualityConfig:
    """质量配置"""
    min_fix_rate: float = 0.95
    max_error_rate: float = 0.05
    require_backup: bool = True
    validate_output: bool = True
    generate_report: bool = True
    report_format: str = "detailed"  # simple, detailed, summary

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = None):
        """初始化配置管理器"""
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 默认配置文件路径
        self.default_config_file = self.config_dir / "default_config.json"
        self.user_config_file = self.config_dir / "user_config.json"
        self.custom_rules_file = self.config_dir / "custom_rules.json"
        
        # 配置对象
        self.fix_rules: List[FixRule] = []
        self.check_rules: List[CheckRule] = []
        self.processing_config = ProcessingConfig()
        self.quality_config = QualityConfig()
        
        # 加载配置
        self._load_default_config()
        self._load_user_config()
        self._load_custom_rules()
    
    def _load_default_config(self):
        """加载默认配置"""
        if self.default_config_file.exists():
            with open(self.default_config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self._parse_config_data(config_data)
        else:
            self._create_default_config()
    
    def _load_user_config(self):
        """加载用户配置"""
        if self.user_config_file.exists():
            with open(self.user_config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self._merge_user_config(user_config)
    
    def _load_custom_rules(self):
        """加载自定义规则"""
        if self.custom_rules_file.exists():
            with open(self.custom_rules_file, 'r', encoding='utf-8') as f:
                custom_rules = json.load(f)
                self._parse_custom_rules(custom_rules)
    
    def _create_default_config(self):
        """创建默认配置"""
        default_config = {
            "fix_rules": [
                {
                    "name": "bracket_mismatch",
                    "description": "修复括号不匹配问题",
                    "enabled": True,
                    "priority": 1,
                    "pattern": r"\$([^$]*?\([^$]*?)\$",
                    "replacement": r"$\1)$",
                    "conditions": {"context": "math_formula"}
                },
                {
                    "name": "greek_letters",
                    "description": "统一希腊字母表示",
                    "enabled": True,
                    "priority": 2,
                    "pattern": r"([αβγδεζηθικλμνξοπρστυφχψω])",
                    "replacement": r"\\\1",
                    "conditions": {"context": "math_formula"}
                },
                {
                    "name": "math_symbols",
                    "description": "统一数学符号表示",
                    "enabled": True,
                    "priority": 3,
                    "pattern": r"([∈⊂⊆≤≥≠×÷±])",
                    "replacement": r"\\\1",
                    "conditions": {"context": "math_formula"}
                }
            ],
            "check_rules": [
                {
                    "name": "unmatched_brackets",
                    "description": "检查未匹配的括号",
                    "enabled": True,
                    "severity": "error",
                    "pattern": r"\$[^$]*?\([^$]*?\$",
                    "message": "发现未匹配的括号",
                    "suggestions": ["添加缺失的右括号", "检查括号嵌套"]
                },
                {
                    "name": "unicode_symbols",
                    "description": "检查Unicode符号使用",
                    "enabled": True,
                    "severity": "warning",
                    "pattern": r"([αβγδεζηθικλμνξοπρστυφχψω])",
                    "message": "发现Unicode符号，建议使用LaTeX命令",
                    "suggestions": ["使用\\alpha代替α", "使用\\beta代替β"]
                }
            ],
            "processing_config": {
                "max_workers": 4,
                "backup_enabled": True,
                "recursive_processing": True,
                "file_patterns": ["*.md"],
                "exclude_patterns": ["*.bak", "*.tmp"],
                "log_level": "INFO",
                "output_format": "json"
            },
            "quality_config": {
                "min_fix_rate": 0.95,
                "max_error_rate": 0.05,
                "require_backup": True,
                "validate_output": True,
                "generate_report": True,
                "report_format": "detailed"
            }
        }
        
        with open(self.default_config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        self._parse_config_data(default_config)
    
    def _parse_config_data(self, config_data: Dict[str, Any]):
        """解析配置数据"""
        # 解析修复规则
        if "fix_rules" in config_data:
            self.fix_rules = []
            for rule_data in config_data["fix_rules"]:
                rule = FixRule(**rule_data)
                self.fix_rules.append(rule)
        
        # 解析检查规则
        if "check_rules" in config_data:
            self.check_rules = []
            for rule_data in config_data["check_rules"]:
                rule = CheckRule(**rule_data)
                self.check_rules.append(rule)
        
        # 解析处理配置
        if "processing_config" in config_data:
            self.processing_config = ProcessingConfig(**config_data["processing_config"])
        
        # 解析质量配置
        if "quality_config" in config_data:
            self.quality_config = QualityConfig(**config_data["quality_config"])
    
    def _merge_user_config(self, user_config: Dict[str, Any]):
        """合并用户配置"""
        # 合并修复规则
        if "fix_rules" in user_config:
            for user_rule in user_config["fix_rules"]:
                # 查找现有规则并更新
                existing_rule = next((r for r in self.fix_rules if r.name == user_rule["name"]), None)
                if existing_rule:
                    for key, value in user_rule.items():
                        setattr(existing_rule, key, value)
                else:
                    # 添加新规则
                    rule = FixRule(**user_rule)
                    self.fix_rules.append(rule)
        
        # 合并检查规则
        if "check_rules" in user_config:
            for user_rule in user_config["check_rules"]:
                existing_rule = next((r for r in self.check_rules if r.name == user_rule["name"]), None)
                if existing_rule:
                    for key, value in user_rule.items():
                        setattr(existing_rule, key, value)
                else:
                    rule = CheckRule(**user_rule)
                    self.check_rules.append(rule)
        
        # 合并处理配置
        if "processing_config" in user_config:
            for key, value in user_config["processing_config"].items():
                if hasattr(self.processing_config, key):
                    setattr(self.processing_config, key, value)
        
        # 合并质量配置
        if "quality_config" in user_config:
            for key, value in user_config["quality_config"].items():
                if hasattr(self.quality_config, key):
                    setattr(self.quality_config, key, value)
    
    def _parse_custom_rules(self, custom_rules: Dict[str, Any]):
        """解析自定义规则"""
        if "fix_rules" in custom_rules:
            for rule_data in custom_rules["fix_rules"]:
                rule = FixRule(**rule_data)
                self.fix_rules.append(rule)
        
        if "check_rules" in custom_rules:
            for rule_data in custom_rules["check_rules"]:
                rule = CheckRule(**rule_data)
                self.check_rules.append(rule)
    
    def save_user_config(self):
        """保存用户配置"""
        user_config = {
            "fix_rules": [asdict(rule) for rule in self.fix_rules],
            "check_rules": [asdict(rule) for rule in self.check_rules],
            "processing_config": asdict(self.processing_config),
            "quality_config": asdict(self.quality_config)
        }
        
        with open(self.user_config_file, 'w', encoding='utf-8') as f:
            json.dump(user_config, f, ensure_ascii=False, indent=2)
    
    def add_fix_rule(self, rule: FixRule):
        """添加修复规则"""
        # 检查是否已存在同名规则
        existing_rule = next((r for r in self.fix_rules if r.name == rule.name), None)
        if existing_rule:
            # 更新现有规则
            for key, value in asdict(rule).items():
                setattr(existing_rule, key, value)
        else:
            # 添加新规则
            self.fix_rules.append(rule)
        
        self.save_user_config()
    
    def add_check_rule(self, rule: CheckRule):
        """添加检查规则"""
        # 检查是否已存在同名规则
        existing_rule = next((r for r in self.check_rules if r.name == rule.name), None)
        if existing_rule:
            # 更新现有规则
            for key, value in asdict(rule).items():
                setattr(existing_rule, key, value)
        else:
            # 添加新规则
            self.check_rules.append(rule)
        
        self.save_user_config()
    
    def remove_rule(self, rule_name: str, rule_type: str = "fix"):
        """删除规则"""
        if rule_type == "fix":
            self.fix_rules = [r for r in self.fix_rules if r.name != rule_name]
        elif rule_type == "check":
            self.check_rules = [r for r in self.check_rules if r.name != rule_name]
        
        self.save_user_config()
    
    def enable_rule(self, rule_name: str, enabled: bool, rule_type: str = "fix"):
        """启用/禁用规则"""
        if rule_type == "fix":
            for rule in self.fix_rules:
                if rule.name == rule_name:
                    rule.enabled = enabled
                    break
        elif rule_type == "check":
            for rule in self.check_rules:
                if rule.name == rule_name:
                    rule.enabled = enabled
                    break
        
        self.save_user_config()
    
    def get_enabled_fix_rules(self) -> List[FixRule]:
        """获取启用的修复规则"""
        return [rule for rule in self.fix_rules if rule.enabled]
    
    def get_enabled_check_rules(self) -> List[CheckRule]:
        """获取启用的检查规则"""
        return [rule for rule in self.check_rules if rule.enabled]
    
    def get_rule_by_name(self, rule_name: str, rule_type: str = "fix") -> Optional[Union[FixRule, CheckRule]]:
        """根据名称获取规则"""
        if rule_type == "fix":
            return next((r for r in self.fix_rules if r.name == rule_name), None)
        elif rule_type == "check":
            return next((r for r in self.check_rules if r.name == rule_name), None)
        return None
    
    def export_config(self, format: str = "json", file_path: str = None) -> str:
        """导出配置"""
        config_data = {
            "fix_rules": [asdict(rule) for rule in self.fix_rules],
            "check_rules": [asdict(rule) for rule in self.check_rules],
            "processing_config": asdict(self.processing_config),
            "quality_config": asdict(self.quality_config),
            "export_time": datetime.now().isoformat()
        }
        
        if format.lower() == "json":
            output = json.dumps(config_data, ensure_ascii=False, indent=2)
        elif format.lower() == "yaml":
            output = yaml.dump(config_data, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(output)
        
        return output
    
    def import_config(self, config_data: Dict[str, Any]):
        """导入配置"""
        self._parse_config_data(config_data)
        self.save_user_config()
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 验证修复规则
        for rule in self.fix_rules:
            if not rule.name:
                errors.append(f"修复规则缺少名称")
            if not rule.pattern:
                errors.append(f"修复规则 {rule.name} 缺少模式")
            if rule.priority < 0:
                errors.append(f"修复规则 {rule.name} 优先级不能为负数")
        
        # 验证检查规则
        for rule in self.check_rules:
            if not rule.name:
                errors.append(f"检查规则缺少名称")
            if not rule.pattern:
                errors.append(f"检查规则 {rule.name} 缺少模式")
            if rule.severity not in ["info", "warning", "error"]:
                errors.append(f"检查规则 {rule.name} 严重程度无效")
        
        # 验证处理配置
        if self.processing_config.max_workers < 1:
            errors.append("最大工作线程数必须大于0")
        if not self.processing_config.file_patterns:
            errors.append("文件模式列表不能为空")
        
        # 验证质量配置
        if not 0 <= self.quality_config.min_fix_rate <= 1:
            errors.append("最小修复率必须在0到1之间")
        if not 0 <= self.quality_config.max_error_rate <= 1:
            errors.append("最大错误率必须在0到1之间")
        
        return errors

class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_regex_pattern(pattern: str) -> bool:
        """验证正则表达式模式"""
        try:
            import re
            re.compile(pattern)
            return True
        except re.error:
            return False
    
    @staticmethod
    def validate_rule_priority(priority: int) -> bool:
        """验证规则优先级"""
        return 0 <= priority <= 100
    
    @staticmethod
    def validate_severity_level(severity: str) -> bool:
        """验证严重程度级别"""
        return severity in ["info", "warning", "error"]

class ConfigTemplate:
    """配置模板"""
    
    @staticmethod
    def get_basic_template() -> Dict[str, Any]:
        """获取基础配置模板"""
        return {
            "fix_rules": [
                {
                    "name": "basic_bracket_fix",
                    "description": "基础括号修复",
                    "enabled": True,
                    "priority": 1,
                    "pattern": r"\$([^$]*?\([^$]*?)\$",
                    "replacement": r"$\1)$",
                    "conditions": {}
                }
            ],
            "check_rules": [
                {
                    "name": "basic_syntax_check",
                    "description": "基础语法检查",
                    "enabled": True,
                    "severity": "warning",
                    "pattern": r"\$[^$]*?\([^$]*?\$",
                    "message": "发现未匹配的括号",
                    "suggestions": ["检查括号匹配"]
                }
            ],
            "processing_config": {
                "max_workers": 4,
                "backup_enabled": True,
                "recursive_processing": True,
                "file_patterns": ["*.md"],
                "exclude_patterns": ["*.bak", "*.tmp"],
                "log_level": "INFO",
                "output_format": "json"
            },
            "quality_config": {
                "min_fix_rate": 0.95,
                "max_error_rate": 0.05,
                "require_backup": True,
                "validate_output": True,
                "generate_report": True,
                "report_format": "detailed"
            }
        }
    
    @staticmethod
    def get_advanced_template() -> Dict[str, Any]:
        """获取高级配置模板"""
        return {
            "fix_rules": [
                {
                    "name": "comprehensive_bracket_fix",
                    "description": "综合括号修复",
                    "enabled": True,
                    "priority": 1,
                    "pattern": r"\$([^$]*?\([^$]*?)\$",
                    "replacement": r"$\1)$",
                    "conditions": {"context": "math_formula"}
                },
                {
                    "name": "greek_letter_standardization",
                    "description": "希腊字母标准化",
                    "enabled": True,
                    "priority": 2,
                    "pattern": r"([αβγδεζηθικλμνξοπρστυφχψω])",
                    "replacement": r"\\\1",
                    "conditions": {"context": "math_formula"}
                },
                {
                    "name": "math_symbol_standardization",
                    "description": "数学符号标准化",
                    "enabled": True,
                    "priority": 3,
                    "pattern": r"([∈⊂⊆≤≥≠×÷±])",
                    "replacement": r"\\\1",
                    "conditions": {"context": "math_formula"}
                }
            ],
            "check_rules": [
                {
                    "name": "comprehensive_syntax_check",
                    "description": "综合语法检查",
                    "enabled": True,
                    "severity": "error",
                    "pattern": r"\$[^$]*?\([^$]*?\$",
                    "message": "发现未匹配的括号",
                    "suggestions": ["添加缺失的右括号", "检查括号嵌套"]
                },
                {
                    "name": "unicode_symbol_check",
                    "description": "Unicode符号检查",
                    "enabled": True,
                    "severity": "warning",
                    "pattern": r"([αβγδεζηθικλμνξοπρστυφχψω])",
                    "message": "发现Unicode符号，建议使用LaTeX命令",
                    "suggestions": ["使用\\alpha代替α", "使用\\beta代替β"]
                },
                {
                    "name": "format_consistency_check",
                    "description": "格式一致性检查",
                    "enabled": True,
                    "severity": "info",
                    "pattern": r"\$\$[^$]*?\$\$",
                    "message": "检查块级公式格式",
                    "suggestions": ["确保公式对齐", "检查换行格式"]
                }
            ],
            "processing_config": {
                "max_workers": 8,
                "backup_enabled": True,
                "recursive_processing": True,
                "file_patterns": ["*.md", "*.markdown"],
                "exclude_patterns": ["*.bak", "*.tmp", "*.old"],
                "log_level": "DEBUG",
                "output_format": "json"
            },
            "quality_config": {
                "min_fix_rate": 0.98,
                "max_error_rate": 0.02,
                "require_backup": True,
                "validate_output": True,
                "generate_report": True,
                "report_format": "detailed"
            }
        }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数学格式修复配置管理器")
    parser.add_argument("--config-dir", help="配置目录路径")
    parser.add_argument("--export", help="导出配置文件路径")
    parser.add_argument("--import", dest="import_file", help="导入配置文件路径")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="导出格式")
    parser.add_argument("--template", choices=["basic", "advanced"], help="使用配置模板")
    parser.add_argument("--validate", action="store_true", help="验证配置")
    parser.add_argument("--list-rules", action="store_true", help="列出所有规则")
    
    args = parser.parse_args()
    
    # 创建配置管理器
    config_manager = ConfigManager(args.config_dir)
    
    if args.import_file:
        # 导入配置
        with open(args.import_file, 'r', encoding='utf-8') as f:
            if args.import_file.endswith('.yaml') or args.import_file.endswith('.yml'):
                import yaml
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        config_manager.import_config(config_data)
        print(f"配置已从 {args.import_file} 导入")
    
    elif args.export:
        # 导出配置
        output = config_manager.export_config(args.format, args.export)
        print(f"配置已导出到 {args.export}")
    
    elif args.template:
        # 使用模板
        if args.template == "basic":
            template = ConfigTemplate.get_basic_template()
        else:
            template = ConfigTemplate.get_advanced_template()
        
        config_manager.import_config(template)
        print(f"已应用 {args.template} 模板配置")
    
    elif args.validate:
        # 验证配置
        errors = config_manager.validate_config()
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("配置验证通过")
    
    elif args.list_rules:
        # 列出规则
        print("修复规则:")
        for rule in config_manager.fix_rules:
            status = "启用" if rule.enabled else "禁用"
            print(f"  - {rule.name}: {rule.description} ({status})")
        
        print("\n检查规则:")
        for rule in config_manager.check_rules:
            status = "启用" if rule.enabled else "禁用"
            print(f"  - {rule.name}: {rule.description} ({status}, {rule.severity})")
    
    else:
        # 显示帮助信息
        print("数学格式修复配置管理器")
        print("=" * 50)
        print(f"配置目录: {config_manager.config_dir}")
        print(f"修复规则数量: {len(config_manager.fix_rules)}")
        print(f"检查规则数量: {len(config_manager.check_rules)}")
        print("\n使用 --help 查看所有选项")

if __name__ == "__main__":
    main() 