#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复命令行工具
提供完整的命令行界面，支持单文件处理、批量处理、配置管理等功能
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# 导入核心修复模块
from 数学格式修复核心模块 import MathFormatFixer
from 数学格式修复配置管理器 import ConfigManager

class MathFormatCLI:
    """数学格式修复命令行工具"""
    
    def __init__(self):
        self.fixer = MathFormatFixer()
        self.config_manager = ConfigManager()
        self.setup_logging()
    
    def setup_logging(self, level=logging.INFO):
        """设置日志"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('math_format_fix.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_single_file(self, file_path: str, output_path: Optional[str] = None, 
                           config_path: Optional[str] = None) -> bool:
        """处理单个文件"""
        try:
            # 加载配置
            if config_path:
                self.config_manager.load_config(config_path)
                self.fixer.update_config(self.config_manager.get_config())
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修复格式
            fixed_content = self.fixer.fix_text(content)
            
            # 确定输出路径
            if not output_path:
                base_name = Path(file_path).stem
                output_path = f"{base_name}_fixed.md"
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            self.logger.info(f"文件处理完成: {file_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"处理文件失败 {file_path}: {str(e)}")
            return False
    
    def process_directory(self, dir_path: str, output_dir: Optional[str] = None,
                         config_path: Optional[str] = None, recursive: bool = False) -> Dict[str, Any]:
        """处理目录中的所有文件"""
        results = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            # 加载配置
            if config_path:
                self.config_manager.load_config(config_path)
                self.fixer.update_config(self.config_manager.get_config())
            
            # 获取所有文件
            files = []
            if recursive:
                for root, dirs, filenames in os.walk(dir_path):
                    for filename in filenames:
                        if filename.endswith('.md'):
                            files.append(os.path.join(root, filename))
            else:
                for filename in os.listdir(dir_path):
                    if filename.endswith('.md'):
                        files.append(os.path.join(dir_path, filename))
            
            results['total_files'] = len(files)
            
            # 创建输出目录
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 处理每个文件
            for file_path in files:
                try:
                    if output_dir:
                        rel_path = os.path.relpath(file_path, dir_path)
                        output_path = os.path.join(output_dir, rel_path)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    else:
                        output_path = None
                    
                    if self.process_single_file(file_path, output_path, config_path):
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(file_path)
                        
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"{file_path}: {str(e)}")
            
            self.logger.info(f"目录处理完成: {results['successful']}/{results['total_files']} 成功")
            
        except Exception as e:
            self.logger.error(f"处理目录失败 {dir_path}: {str(e)}")
            results['errors'].append(f"目录错误: {str(e)}")
        
        return results
    
    def check_format(self, file_path: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """检查文件格式"""
        try:
            # 加载配置
            if config_path:
                self.config_manager.load_config(config_path)
                self.fixer.update_config(self.config_manager.get_config())
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查格式
            issues = self.fixer.check_format(content)
            
            return {
                'file': file_path,
                'total_issues': len(issues),
                'issues': issues,
                'status': 'OK' if len(issues) == 0 else 'NEEDS_FIX'
            }
            
        except Exception as e:
            return {
                'file': file_path,
                'error': str(e),
                'status': 'ERROR'
            }
    
    def generate_report(self, results: Dict[str, Any], output_file: str):
        """生成处理报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_files': results.get('total_files', 0),
                'successful': results.get('successful', 0),
                'failed': results.get('failed', 0),
                'success_rate': f"{results.get('successful', 0) / max(results.get('total_files', 1), 1) * 100:.1f}%"
            },
            'errors': results.get('errors', []),
            'details': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"报告已生成: {output_file}")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
数学格式修复命令行工具

用法:
  python 数学格式修复命令行工具.py <命令> [选项]

命令:
  fix-file <文件路径>          修复单个文件
  fix-dir <目录路径>           修复目录中的所有文件
  check <文件路径>             检查文件格式问题
  config <配置文件路径>         验证配置文件
  help                        显示此帮助信息

选项:
  -o, --output <路径>          指定输出路径
  -c, --config <配置文件>       指定配置文件
  -r, --recursive             递归处理子目录
  -v, --verbose               详细输出
  --report <报告文件>          生成处理报告

示例:
  python 数学格式修复命令行工具.py fix-file input.md -o output.md
  python 数学格式修复命令行工具.py fix-dir ./docs -r -c config.json
  python 数学格式修复命令行工具.py check input.md
        """
        print(help_text)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='数学格式修复命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # fix-file 命令
    fix_file_parser = subparsers.add_parser('fix-file', help='修复单个文件')
    fix_file_parser.add_argument('file', help='输入文件路径')
    fix_file_parser.add_argument('-o', '--output', help='输出文件路径')
    fix_file_parser.add_argument('-c', '--config', help='配置文件路径')
    
    # fix-dir 命令
    fix_dir_parser = subparsers.add_parser('fix-dir', help='修复目录中的所有文件')
    fix_dir_parser.add_argument('directory', help='输入目录路径')
    fix_dir_parser.add_argument('-o', '--output', help='输出目录路径')
    fix_dir_parser.add_argument('-c', '--config', help='配置文件路径')
    fix_dir_parser.add_argument('-r', '--recursive', action='store_true', help='递归处理子目录')
    
    # check 命令
    check_parser = subparsers.add_parser('check', help='检查文件格式问题')
    check_parser.add_argument('file', help='要检查的文件路径')
    check_parser.add_argument('-c', '--config', help='配置文件路径')
    
    # config 命令
    config_parser = subparsers.add_parser('config', help='验证配置文件')
    config_parser.add_argument('config', help='配置文件路径')
    
    # 全局选项
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    parser.add_argument('--report', help='生成处理报告')
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # 创建CLI工具
    cli = MathFormatCLI()
    cli.setup_logging(log_level)
    
    try:
        if args.command == 'fix-file':
            success = cli.process_single_file(args.file, args.output, args.config)
            if args.report:
                cli.generate_report({'successful': 1 if success else 0, 'total_files': 1}, args.report)
            sys.exit(0 if success else 1)
            
        elif args.command == 'fix-dir':
            results = cli.process_directory(args.directory, args.output, args.config, args.recursive)
            if args.report:
                cli.generate_report(results, args.report)
            sys.exit(0 if results['failed'] == 0 else 1)
            
        elif args.command == 'check':
            result = cli.check_format(args.file, args.config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            sys.exit(0 if result['status'] == 'OK' else 1)
            
        elif args.command == 'config':
            try:
                cli.config_manager.load_config(args.config)
                print("配置文件验证成功")
                print(json.dumps(cli.config_manager.get_config(), ensure_ascii=False, indent=2))
                sys.exit(0)
            except Exception as e:
                print(f"配置文件验证失败: {str(e)}")
                sys.exit(1)
                
        else:
            cli.show_help()
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        cli.logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 