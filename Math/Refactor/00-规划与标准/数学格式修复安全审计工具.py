#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复安全审计工具
提供安全漏洞检测、权限验证和安全建议
"""

import os
import json
import hashlib
import logging
import re
import ast
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import subprocess
import platform

@dataclass
class SecurityIssue:
    """安全问题数据类"""
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    category: str  # 'input_validation', 'authentication', 'authorization', 'data_protection', etc.
    title: str
    description: str
    location: str
    line_number: Optional[int]
    recommendation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None

@dataclass
class SecurityAuditResult:
    """安全审计结果"""
    timestamp: datetime
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    info_issues: int
    security_score: float
    issues: List[SecurityIssue]
    recommendations: List[str]

class SecurityAuditor:
    """安全审计器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.known_vulnerabilities = self._load_known_vulnerabilities()
        self.security_patterns = self._load_security_patterns()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('SecurityAuditor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_known_vulnerabilities(self) -> Dict[str, Any]:
        """加载已知漏洞数据库"""
        return {
            'sql_injection': {
                'patterns': [r'execute\(.*\+.*\)', r'cursor\.execute\(.*\+.*\)'],
                'severity': 'critical',
                'cwe_id': 'CWE-89',
                'description': 'SQL注入漏洞'
            },
            'xss': {
                'patterns': [r'innerHTML\s*=', r'outerHTML\s*=', r'document\.write\('],
                'severity': 'high',
                'cwe_id': 'CWE-79',
                'description': '跨站脚本攻击'
            },
            'path_traversal': {
                'patterns': [r'open\(.*\.\./', r'Path\(.*\.\./'],
                'severity': 'high',
                'cwe_id': 'CWE-22',
                'description': '路径遍历漏洞'
            },
            'command_injection': {
                'patterns': [r'os\.system\(', r'subprocess\.call\(', r'eval\('],
                'severity': 'critical',
                'cwe_id': 'CWE-78',
                'description': '命令注入漏洞'
            },
            'weak_crypto': {
                'patterns': [r'md5\(', r'sha1\(', r'base64\.encode\('],
                'severity': 'medium',
                'cwe_id': 'CWE-327',
                'description': '弱加密算法'
            }
        }
    
    def _load_security_patterns(self) -> Dict[str, Any]:
        """加载安全模式"""
        return {
            'input_validation': {
                'required': ['re.match', 'isinstance', 'validation'],
                'severity': 'medium'
            },
            'authentication': {
                'required': ['login', 'auth', 'session'],
                'severity': 'high'
            },
            'authorization': {
                'required': ['permission', 'role', 'access'],
                'severity': 'high'
            },
            'data_protection': {
                'required': ['encrypt', 'hash', 'secure'],
                'severity': 'medium'
            },
            'logging': {
                'required': ['logging', 'log'],
                'severity': 'low'
            }
        }
    
    def audit_file(self, file_path: str) -> List[SecurityIssue]:
        """审计单个文件"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # 检查已知漏洞
            for vuln_name, vuln_info in self.known_vulnerabilities.items():
                for pattern in vuln_info['patterns']:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        issues.append(SecurityIssue(
                            severity=vuln_info['severity'],
                            category='known_vulnerability',
                            title=f"发现{vuln_info['description']}",
                            description=f"在{file_path}第{line_number}行发现{vuln_info['description']}",
                            location=file_path,
                            line_number=line_number,
                            recommendation=f"建议修复{vuln_info['description']}",
                            cwe_id=vuln_info['cwe_id']
                        ))
            
            # 检查安全模式
            for pattern_name, pattern_info in self.security_patterns.items():
                found_patterns = []
                for required in pattern_info['required']:
                    if required.lower() in content.lower():
                        found_patterns.append(required)
                
                if not found_patterns:
                    issues.append(SecurityIssue(
                        severity=pattern_info['severity'],
                        category='missing_security_pattern',
                        title=f"缺少{pattern_name}安全模式",
                        description=f"文件{file_path}缺少{pattern_name}相关的安全实现",
                        location=file_path,
                        line_number=None,
                        recommendation=f"建议添加{pattern_name}相关的安全实现"
                    ))
            
            # 检查Python特定安全问题
            python_issues = self._check_python_specific_issues(content, file_path)
            issues.extend(python_issues)
            
        except Exception as e:
            self.logger.error(f"审计文件{file_path}时出错: {str(e)}")
            issues.append(SecurityIssue(
                severity='medium',
                category='audit_error',
                title='文件审计错误',
                description=f'无法审计文件{file_path}: {str(e)}',
                location=file_path,
                line_number=None,
                recommendation='检查文件权限和格式'
            ))
        
        return issues
    
    def _check_python_specific_issues(self, content: str, file_path: str) -> List[SecurityIssue]:
        """检查Python特定的安全问题"""
        issues = []
        
        try:
            # 解析AST
            tree = ast.parse(content)
            
            # 检查危险函数调用
            dangerous_functions = {
                'eval': 'eval函数可能导致代码注入',
                'exec': 'exec函数可能导致代码注入',
                'input': 'input函数可能接收恶意输入',
                'pickle.loads': 'pickle可能执行恶意代码',
                'marshal.loads': 'marshal可能执行恶意代码'
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in dangerous_functions:
                            line_number = node.lineno
                            issues.append(SecurityIssue(
                                severity='critical',
                                category='dangerous_function',
                                title=f'使用危险函数{func_name}',
                                description=f'在{file_path}第{line_number}行使用了{dangerous_functions[func_name]}',
                                location=file_path,
                                line_number=line_number,
                                recommendation=f'建议替换{func_name}为更安全的替代方案'
                            ))
                    
                    elif isinstance(node.func, ast.Attribute):
                        if hasattr(node.func, 'attr'):
                            attr_name = node.func.attr
                            if attr_name in dangerous_functions:
                                line_number = node.lineno
                                issues.append(SecurityIssue(
                                    severity='critical',
                                    category='dangerous_function',
                                    title=f'使用危险函数{attr_name}',
                                    description=f'在{file_path}第{line_number}行使用了{dangerous_functions[attr_name]}',
                                    location=file_path,
                                    line_number=line_number,
                                    recommendation=f'建议替换{attr_name}为更安全的替代方案'
                                ))
            
            # 检查硬编码敏感信息
            sensitive_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', '硬编码密码'),
                (r'api_key\s*=\s*["\'][^"\']+["\']', '硬编码API密钥'),
                (r'secret\s*=\s*["\'][^"\']+["\']', '硬编码密钥'),
                (r'token\s*=\s*["\'][^"\']+["\']', '硬编码令牌')
            ]
            
            for pattern, description in sensitive_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_number = content[:match.start()].count('\n') + 1
                    issues.append(SecurityIssue(
                        severity='high',
                        category='hardcoded_credentials',
                        title=description,
                        description=f'在{file_path}第{line_number}行发现{description}',
                        location=file_path,
                        line_number=line_number,
                        recommendation='建议使用环境变量或配置文件存储敏感信息'
                    ))
            
        except SyntaxError as e:
            issues.append(SecurityIssue(
                severity='low',
                category='syntax_error',
                title='Python语法错误',
                description=f'文件{file_path}存在语法错误: {str(e)}',
                location=file_path,
                line_number=None,
                recommendation='修复Python语法错误'
            ))
        except Exception as e:
            self.logger.error(f"检查Python特定问题时出错: {str(e)}")
        
        return issues
    
    def audit_directory(self, directory_path: str) -> SecurityAuditResult:
        """审计整个目录"""
        all_issues = []
        
        # 遍历目录中的所有Python文件
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    issues = self.audit_file(file_path)
                    all_issues.extend(issues)
        
        # 统计问题
        critical_issues = len([i for i in all_issues if i.severity == 'critical'])
        high_issues = len([i for i in all_issues if i.severity == 'high'])
        medium_issues = len([i for i in all_issues if i.severity == 'medium'])
        low_issues = len([i for i in all_issues if i.severity == 'low'])
        info_issues = len([i for i in all_issues if i.severity == 'info'])
        
        # 计算安全评分 (0-100)
        total_issues = len(all_issues)
        security_score = 100.0
        
        if total_issues > 0:
            # 根据问题严重程度扣分
            security_score -= critical_issues * 20
            security_score -= high_issues * 10
            security_score -= medium_issues * 5
            security_score -= low_issues * 2
            security_score -= info_issues * 1
            
            security_score = max(0, security_score)
        
        # 生成建议
        recommendations = self._generate_security_recommendations(all_issues)
        
        return SecurityAuditResult(
            timestamp=datetime.now(),
            total_issues=total_issues,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            info_issues=info_issues,
            security_score=security_score,
            issues=all_issues,
            recommendations=recommendations
        )
    
    def _generate_security_recommendations(self, issues: List[SecurityIssue]) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        # 按类别统计问题
        categories = {}
        for issue in issues:
            if issue.category not in categories:
                categories[issue.category] = []
            categories[issue.category].append(issue)
        
        # 生成针对性建议
        if 'known_vulnerability' in categories:
            recommendations.append("立即修复已知安全漏洞，特别是SQL注入和命令注入")
        
        if 'dangerous_function' in categories:
            recommendations.append("替换危险函数调用，使用更安全的替代方案")
        
        if 'hardcoded_credentials' in categories:
            recommendations.append("移除硬编码的敏感信息，使用环境变量或配置文件")
        
        if 'missing_security_pattern' in categories:
            recommendations.append("添加必要的安全模式实现，如输入验证和身份验证")
        
        if len(issues) == 0:
            recommendations.append("未发现安全问题，继续保持良好的安全实践")
        
        # 通用建议
        recommendations.extend([
            "定期进行安全审计",
            "保持依赖包的最新版本",
            "实施安全编码规范",
            "建立安全测试流程"
        ])
        
        return recommendations
    
    def generate_security_report(self, result: SecurityAuditResult, output_file: str = None) -> str:
        """生成安全审计报告"""
        report = f"""
数学格式修复项目安全审计报告
生成时间: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

安全评分: {result.security_score:.1f}/100

问题统计:
  严重问题: {result.critical_issues}
  高危问题: {result.high_issues}
  中危问题: {result.medium_issues}
  低危问题: {result.low_issues}
  信息问题: {result.info_issues}
  总计: {result.total_issues}

详细问题列表:
"""
        
        # 按严重程度分组显示问题
        severity_order = ['critical', 'high', 'medium', 'low', 'info']
        
        for severity in severity_order:
            severity_issues = [i for i in result.issues if i.severity == severity]
            if severity_issues:
                report += f"\n{severity.upper()} 级别问题:\n"
                for issue in severity_issues:
                    report += f"  - {issue.title}\n"
                    report += f"    位置: {issue.location}"
                    if issue.line_number:
                        report += f":{issue.line_number}"
                    report += f"\n    描述: {issue.description}\n"
                    report += f"    建议: {issue.recommendation}\n"
        
        # 安全建议
        report += f"\n安全建议:\n"
        for i, recommendation in enumerate(result.recommendations, 1):
            report += f"  {i}. {recommendation}\n"
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"安全审计报告已保存: {output_file}")
        
        return report
    
    def check_dependencies(self, requirements_file: str = 'requirements.txt') -> List[Dict[str, Any]]:
        """检查依赖包的安全漏洞"""
        vulnerabilities = []
        
        if not os.path.exists(requirements_file):
            self.logger.warning(f"依赖文件{requirements_file}不存在")
            return vulnerabilities
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.read().splitlines()
            
            for requirement in requirements:
                if requirement.strip() and not requirement.startswith('#'):
                    # 解析包名和版本
                    package_info = self._parse_requirement(requirement)
                    if package_info:
                        vuln_info = self._check_package_vulnerability(package_info)
                        if vuln_info:
                            vulnerabilities.append(vuln_info)
        
        except Exception as e:
            self.logger.error(f"检查依赖包时出错: {str(e)}")
        
        return vulnerabilities
    
    def _parse_requirement(self, requirement: str) -> Optional[Dict[str, str]]:
        """解析依赖包信息"""
        try:
            # 简单的包名和版本解析
            if '==' in requirement:
                package, version = requirement.split('==', 1)
            elif '>=' in requirement:
                package, version = requirement.split('>=', 1)
            elif '<=' in requirement:
                package, version = requirement.split('<=', 1)
            else:
                package = requirement.strip()
                version = 'latest'
            
            return {
                'package': package.strip(),
                'version': version.strip()
            }
        except:
            return None
    
    def _check_package_vulnerability(self, package_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """检查包的安全漏洞（模拟）"""
        # 这里应该连接到真实的漏洞数据库
        # 目前使用模拟数据
        known_vulnerable_packages = {
            'flask': {
                'versions': ['<2.0.0'],
                'vulnerabilities': ['CVE-2021-23336', 'CVE-2020-26160']
            },
            'requests': {
                'versions': ['<2.25.0'],
                'vulnerabilities': ['CVE-2021-33503']
            }
        }
        
        package_name = package_info['package'].lower()
        if package_name in known_vulnerable_packages:
            return {
                'package': package_name,
                'version': package_info['version'],
                'vulnerabilities': known_vulnerable_packages[package_name]['vulnerabilities'],
                'severity': 'high'
            }
        
        return None
    
    def export_audit_results(self, result: SecurityAuditResult, output_file: str):
        """导出审计结果"""
        export_data = {
            'timestamp': result.timestamp.isoformat(),
            'security_score': result.security_score,
            'statistics': {
                'total_issues': result.total_issues,
                'critical_issues': result.critical_issues,
                'high_issues': result.high_issues,
                'medium_issues': result.medium_issues,
                'low_issues': result.low_issues,
                'info_issues': result.info_issues
            },
            'issues': [asdict(issue) for issue in result.issues],
            'recommendations': result.recommendations
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"审计结果已导出: {output_file}")

class SecurityValidator:
    """安全验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger('SecurityValidator')
    
    def validate_input(self, input_data: str) -> Tuple[bool, List[str]]:
        """验证输入数据的安全性"""
        issues = []
        
        # 检查SQL注入
        sql_patterns = [r'(\b(union|select|insert|update|delete|drop|create)\b)', r'(\b(or|and)\b\s+\d+\s*=\s*\d+)']
        for pattern in sql_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                issues.append("检测到可能的SQL注入")
        
        # 检查XSS
        xss_patterns = [r'<script[^>]*>', r'javascript:', r'on\w+\s*=']
        for pattern in xss_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                issues.append("检测到可能的XSS攻击")
        
        # 检查命令注入
        cmd_patterns = [r'(\b(cat|ls|rm|del|format|shutdown)\b)', r'(\||&|;|`|$)']
        for pattern in cmd_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                issues.append("检测到可能的命令注入")
        
        # 检查路径遍历
        path_patterns = [r'\.\./', r'\.\.\\']
        for pattern in path_patterns:
            if pattern in input_data:
                issues.append("检测到可能的路径遍历")
        
        return len(issues) == 0, issues
    
    def validate_file_permissions(self, file_path: str) -> Tuple[bool, List[str]]:
        """验证文件权限"""
        issues = []
        
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                issues.append("文件不存在")
                return False, issues
            
            # 检查文件权限
            stat_info = os.stat(file_path)
            
            # 检查是否所有用户都可写
            if stat_info.st_mode & 0o777 == 0o777:
                issues.append("文件权限过于开放")
            
            # 检查所有者
            if stat_info.st_uid == 0:  # root用户
                issues.append("文件所有者是root用户")
            
        except Exception as e:
            issues.append(f"无法检查文件权限: {str(e)}")
        
        return len(issues) == 0, issues
    
    def validate_environment(self) -> Dict[str, Any]:
        """验证运行环境的安全性"""
        env_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'user': os.getenv('USER', 'unknown'),
            'home': os.getenv('HOME', 'unknown'),
            'security_issues': []
        }
        
        # 检查是否以root权限运行
        if os.geteuid() == 0:
            env_info['security_issues'].append("以root权限运行")
        
        # 检查环境变量
        sensitive_vars = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']
        for var in sensitive_vars:
            if os.getenv(var):
                env_info['security_issues'].append(f"发现敏感环境变量: {var}")
        
        return env_info

def main():
    """主函数 - 演示安全审计工具"""
    auditor = SecurityAuditor()
    validator = SecurityValidator()
    
    print("开始安全审计...")
    
    # 审计当前目录
    current_dir = os.getcwd()
    result = auditor.audit_directory(current_dir)
    
    # 生成报告
    report = auditor.generate_security_report(result, "security_audit_report.txt")
    print(report)
    
    # 导出结果
    auditor.export_audit_results(result, "security_audit_results.json")
    
    # 检查依赖包
    print("\n检查依赖包安全漏洞...")
    vulnerabilities = auditor.check_dependencies()
    if vulnerabilities:
        print("发现依赖包安全漏洞:")
        for vuln in vulnerabilities:
            print(f"  - {vuln['package']} {vuln['version']}: {vuln['vulnerabilities']}")
    else:
        print("未发现依赖包安全漏洞")
    
    # 验证环境
    print("\n验证运行环境...")
    env_info = validator.validate_environment()
    print(f"Python版本: {env_info['python_version']}")
    print(f"平台: {env_info['platform']}")
    if env_info['security_issues']:
        print("环境安全问题:")
        for issue in env_info['security_issues']:
            print(f"  - {issue}")
    else:
        print("环境安全检查通过")
    
    print("\n安全审计完成!")

if __name__ == '__main__':
    main() 