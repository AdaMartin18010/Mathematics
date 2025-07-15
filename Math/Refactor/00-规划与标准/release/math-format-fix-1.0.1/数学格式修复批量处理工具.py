#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复批量处理工具
Math/Refactor项目数学格式修复批量处理工具

作者: 数学知识体系重构项目组
时间: 2025年1月
版本: 1.0
"""

import os
import re
import time
import json
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from tqdm import tqdm

# 导入修复工具
import sys
sys.path.append(str(Path(__file__).parent))
from 数学格式修复执行脚本 import MathFormatFixer, MathFormatChecker

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, max_workers: int = 4, log_level: str = "INFO"):
        """初始化批量处理器"""
        self.max_workers = max_workers
        self.fixer = MathFormatFixer("batch")
        self.checker = MathFormatChecker("batch")
        
        # 设置日志
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('batch_processing.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_errors_fixed': 0,
            'total_errors_found': 0,
            'processing_time': 0,
            'start_time': None,
            'end_time': None
        }
        
        # 错误统计
        self.error_stats = {
            'syntax_errors': 0,
            'symbol_errors': 0,
            'format_errors': 0,
            'other_errors': 0
        }
    
    def process_directory(self, directory_path: str, recursive: bool = True, 
                         file_pattern: str = "*.md", backup: bool = True) -> Dict[str, Any]:
        """处理目录中的所有文件"""
        self.stats['start_time'] = datetime.now()
        self.logger.info(f"开始处理目录: {directory_path}")
        
        # 获取所有文件
        files = self._get_files(directory_path, recursive, file_pattern)
        self.stats['total_files'] = len(files)
        
        if not files:
            self.logger.warning(f"在目录 {directory_path} 中没有找到匹配的文件")
            return self._get_summary()
        
        self.logger.info(f"找到 {len(files)} 个文件需要处理")
        
        # 创建备份目录
        if backup:
            backup_dir = self._create_backup_directory(directory_path)
        
        # 批量处理文件
        results = []
        with tqdm(total=len(files), desc="处理文件") as pbar:
            if self.max_workers > 1:
                results = self._process_files_parallel(files, backup, pbar)
            else:
                results = self._process_files_sequential(files, backup, pbar)
        
        # 更新统计信息
        self._update_stats(results)
        
        # 生成报告
        self.stats['end_time'] = datetime.now()
        self.stats['processing_time'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        summary = self._get_summary()
        self._save_report(summary, directory_path)
        
        self.logger.info(f"批量处理完成，共处理 {self.stats['processed_files']} 个文件")
        return summary
    
    def _get_files(self, directory_path: str, recursive: bool, file_pattern: str) -> List[Path]:
        """获取需要处理的文件列表"""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        if recursive:
            files = list(directory.rglob(file_pattern))
        else:
            files = list(directory.glob(file_pattern))
        
        return files
    
    def _create_backup_directory(self, directory_path: str) -> Path:
        """创建备份目录"""
        backup_dir = Path(directory_path) / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)
        self.logger.info(f"创建备份目录: {backup_dir}")
        return backup_dir
    
    def _process_files_parallel(self, files: List[Path], backup: bool, pbar: tqdm) -> List[Dict[str, Any]]:
        """并行处理文件"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self._process_single_file, file, backup): file 
                for file in files
            }
            
            # 收集结果
            for future in future_to_file:
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    file = future_to_file[future]
                    self.logger.error(f"处理文件 {file} 时发生错误: {e}")
                    results.append({
                        'file': str(file),
                        'status': 'failed',
                        'error': str(e),
                        'errors_fixed': 0,
                        'errors_found': 0
                    })
                    pbar.update(1)
        
        return results
    
    def _process_files_sequential(self, files: List[Path], backup: bool, pbar: tqdm) -> List[Dict[str, Any]]:
        """顺序处理文件"""
        results = []
        
        for file in files:
            try:
                result = self._process_single_file(file, backup)
                results.append(result)
            except Exception as e:
                self.logger.error(f"处理文件 {file} 时发生错误: {e}")
                results.append({
                    'file': str(file),
                    'status': 'failed',
                    'error': str(e),
                    'errors_fixed': 0,
                    'errors_found': 0
                })
            pbar.update(1)
        
        return results
    
    def _process_single_file(self, file_path: Path, backup: bool) -> Dict[str, Any]:
        """处理单个文件"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 创建备份
            if backup:
                backup_path = file_path.parent / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / file_path.name
                backup_path.parent.mkdir(exist_ok=True)
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # 检查原始错误
            original_errors = self.checker.check_math_syntax(content)
            original_error_count = len(original_errors)
            
            # 修复错误
            fixed_content = content
            fixed_content = self.fixer.fix_bracket_mismatch(fixed_content)
            fixed_content = self.fixer.fix_greek_letters(fixed_content)
            fixed_content = self.fixer.fix_math_symbols(fixed_content)
            fixed_content = self.fixer.fix_inline_formulas(fixed_content)
            fixed_content = self.fixer.fix_block_formulas(fixed_content)
            fixed_content = self.fixer.fix_alignment(fixed_content)
            
            # 检查修复后的错误
            final_errors = self.checker.check_math_syntax(fixed_content)
            final_error_count = len(final_errors)
            
            # 计算修复的错误数量
            errors_fixed = original_error_count - final_error_count
            
            # 保存修复后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            return {
                'file': str(file_path),
                'status': 'success',
                'original_errors': original_error_count,
                'final_errors': final_error_count,
                'errors_fixed': errors_fixed,
                'errors_found': original_error_count,
                'backup_created': backup
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'status': 'failed',
                'error': str(e),
                'errors_fixed': 0,
                'errors_found': 0
            }
    
    def _update_stats(self, results: List[Dict[str, Any]]):
        """更新统计信息"""
        for result in results:
            self.stats['processed_files'] += 1
            
            if result['status'] == 'success':
                self.stats['successful_files'] += 1
                self.stats['total_errors_fixed'] += result['errors_fixed']
                self.stats['total_errors_found'] += result['errors_found']
            else:
                self.stats['failed_files'] += 1
    
    def _get_summary(self) -> Dict[str, Any]:
        """获取处理摘要"""
        success_rate = (self.stats['successful_files'] / self.stats['total_files'] * 100) if self.stats['total_files'] > 0 else 0
        fix_rate = (self.stats['total_errors_fixed'] / self.stats['total_errors_found'] * 100) if self.stats['total_errors_found'] > 0 else 0
        
        return {
            'summary': {
                'total_files': self.stats['total_files'],
                'processed_files': self.stats['processed_files'],
                'successful_files': self.stats['successful_files'],
                'failed_files': self.stats['failed_files'],
                'success_rate': f"{success_rate:.2f}%",
                'total_errors_found': self.stats['total_errors_found'],
                'total_errors_fixed': self.stats['total_errors_fixed'],
                'fix_rate': f"{fix_rate:.2f}%",
                'processing_time': f"{self.stats['processing_time']:.2f}秒",
                'average_time_per_file': f"{self.stats['processing_time'] / self.stats['total_files']:.3f}秒" if self.stats['total_files'] > 0 else "0秒"
            },
            'error_stats': self.error_stats,
            'timing': {
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                'end_time': self.stats['end_time'].isoformat() if self.stats['end_time'] else None,
                'duration': f"{self.stats['processing_time']:.2f}秒"
            }
        }
    
    def _save_report(self, summary: Dict[str, Any], directory_path: str):
        """保存处理报告"""
        report_path = Path(directory_path) / f"batch_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"处理报告已保存到: {report_path}")

class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self):
        """初始化进度监控器"""
        self.start_time = None
        self.current_file = None
        self.processed_files = 0
        self.total_files = 0
        self.errors_fixed = 0
        self.errors_found = 0
    
    def start_monitoring(self, total_files: int):
        """开始监控"""
        self.start_time = datetime.now()
        self.total_files = total_files
        self.processed_files = 0
        self.errors_fixed = 0
        self.errors_found = 0
    
    def update_progress(self, file_path: str, errors_fixed: int, errors_found: int):
        """更新进度"""
        self.current_file = file_path
        self.processed_files += 1
        self.errors_fixed += errors_fixed
        self.errors_found += errors_found
    
    def get_progress_info(self) -> Dict[str, Any]:
        """获取进度信息"""
        if not self.start_time:
            return {}
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        progress_percentage = (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0
        
        if self.processed_files > 0:
            estimated_total_time = elapsed_time * self.total_files / self.processed_files
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        return {
            'progress_percentage': f"{progress_percentage:.1f}%",
            'processed_files': self.processed_files,
            'total_files': self.total_files,
            'current_file': self.current_file,
            'elapsed_time': f"{elapsed_time:.1f}秒",
            'remaining_time': f"{remaining_time:.1f}秒",
            'errors_fixed': self.errors_fixed,
            'errors_found': self.errors_found,
            'fix_rate': f"{(self.errors_fixed / self.errors_found * 100) if self.errors_found > 0 else 0:.1f}%"
        }

class BatchProcessorGUI:
    """批量处理器图形界面"""
    
    def __init__(self):
        """初始化GUI"""
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
            self.tk = tk
            self.ttk = ttk
            self.filedialog = filedialog
            self.messagebox = messagebox
            self.gui_available = True
        except ImportError:
            self.gui_available = False
            print("GUI不可用，请安装tkinter")
    
    def create_gui(self):
        """创建图形界面"""
        if not self.gui_available:
            return
        
        self.root = self.tk.Tk()
        self.root.title("数学格式修复批量处理工具")
        self.root.geometry("600x400")
        
        # 创建主框架
        main_frame = self.ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(self.tk.W, self.tk.E, self.tk.N, self.tk.S))
        
        # 目录选择
        self.ttk.Label(main_frame, text="选择目录:").grid(row=0, column=0, sticky=self.tk.W)
        self.directory_var = self.tk.StringVar()
        self.ttk.Entry(main_frame, textvariable=self.directory_var, width=50).grid(row=0, column=1, padx=5)
        self.ttk.Button(main_frame, text="浏览", command=self._browse_directory).grid(row=0, column=2)
        
        # 选项设置
        options_frame = self.ttk.LabelFrame(main_frame, text="处理选项", padding="5")
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=10)
        
        self.recursive_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(options_frame, text="递归处理子目录", variable=self.recursive_var).grid(row=0, column=0, sticky=self.tk.W)
        
        self.backup_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(options_frame, text="创建备份", variable=self.backup_var).grid(row=0, column=1, sticky=self.tk.W)
        
        self.ttk.Label(options_frame, text="最大线程数:").grid(row=1, column=0, sticky=self.tk.W)
        self.workers_var = self.tk.StringVar(value="4")
        self.ttk.Spinbox(options_frame, from_=1, to=16, textvariable=self.workers_var, width=10).grid(row=1, column=1, sticky=self.tk.W)
        
        # 进度条
        self.ttk.Label(main_frame, text="处理进度:").grid(row=2, column=0, sticky=self.tk.W, pady=(10, 0))
        self.progress_var = self.tk.StringVar(value="准备就绪")
        self.ttk.Label(main_frame, textvariable=self.progress_var).grid(row=2, column=1, sticky=self.tk.W, pady=(10, 0))
        
        self.progress_bar = self.ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E), pady=5)
        
        # 日志显示
        self.ttk.Label(main_frame, text="处理日志:").grid(row=4, column=0, sticky=self.tk.W, pady=(10, 0))
        
        log_frame = self.ttk.Frame(main_frame)
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(self.tk.W, self.tk.E, self.tk.N, self.tk.S), pady=5)
        
        self.log_text = self.tk.Text(log_frame, height=10, width=70)
        scrollbar = self.ttk.Scrollbar(log_frame, orient=self.tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(self.tk.W, self.tk.E, self.tk.N, self.tk.S))
        scrollbar.grid(row=0, column=1, sticky=(self.tk.N, self.tk.S))
        
        # 按钮
        button_frame = self.ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)
        
        self.start_button = self.ttk.Button(button_frame, text="开始处理", command=self._start_processing)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = self.ttk.Button(button_frame, text="停止处理", command=self._stop_processing, state=self.tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.ttk.Button(button_frame, text="退出", command=self.root.quit).grid(row=0, column=2, padx=5)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def _browse_directory(self):
        """浏览目录"""
        directory = self.filedialog.askdirectory()
        if directory:
            self.directory_var.set(directory)
    
    def _start_processing(self):
        """开始处理"""
        directory = self.directory_var.get()
        if not directory:
            self.messagebox.showerror("错误", "请选择要处理的目录")
            return
        
        # 禁用开始按钮，启用停止按钮
        self.start_button.config(state=self.tk.DISABLED)
        self.stop_button.config(state=self.tk.NORMAL)
        
        # 清空日志
        self.log_text.delete(1.0, self.tk.END)
        
        # 在新线程中运行处理
        import threading
        self.processing_thread = threading.Thread(target=self._run_processing, args=(directory,))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _run_processing(self, directory):
        """运行处理"""
        try:
            processor = BatchProcessor(
                max_workers=int(self.workers_var.get()),
                log_level="INFO"
            )
            
            # 重定向日志到GUI
            class GUILogHandler(logging.Handler):
                def __init__(self, text_widget):
                    super().__init__()
                    self.text_widget = text_widget
                
                def emit(self, record):
                    msg = self.format(record)
                    self.text_widget.insert(self.tk.END, msg + '\n')
                    self.text_widget.see(self.tk.END)
                    self.text_widget.update()
            
            gui_handler = GUILogHandler(self.log_text)
            gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            processor.logger.addHandler(gui_handler)
            
            # 处理目录
            summary = processor.process_directory(
                directory,
                recursive=self.recursive_var.get(),
                backup=self.backup_var.get()
            )
            
            # 显示结果
            self.progress_var.set("处理完成")
            self.progress_bar['value'] = 100
            
            result_msg = f"""
处理完成！

总文件数: {summary['summary']['total_files']}
成功处理: {summary['summary']['successful_files']}
失败文件: {summary['summary']['failed_files']}
成功率: {summary['summary']['success_rate']}
修复错误: {summary['summary']['total_errors_fixed']}
发现错误: {summary['summary']['total_errors_found']}
修复率: {summary['summary']['fix_rate']}
处理时间: {summary['summary']['processing_time']}
            """
            
            self.messagebox.showinfo("处理完成", result_msg)
            
        except Exception as e:
            self.messagebox.showerror("错误", f"处理过程中发生错误: {e}")
        finally:
            # 恢复按钮状态
            self.start_button.config(state=self.tk.NORMAL)
            self.stop_button.config(state=self.tk.DISABLED)
    
    def _stop_processing(self):
        """停止处理"""
        # 这里可以实现停止处理的逻辑
        self.messagebox.showinfo("提示", "停止处理功能待实现")
    
    def run(self):
        """运行GUI"""
        if self.gui_available:
            self.create_gui()
            self.root.mainloop()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数学格式修复批量处理工具")
    parser.add_argument("--directory", "-d", help="要处理的目录路径")
    parser.add_argument("--recursive", "-r", action="store_true", default=True, help="递归处理子目录")
    parser.add_argument("--backup", "-b", action="store_true", default=True, help="创建备份")
    parser.add_argument("--workers", "-w", type=int, default=4, help="最大线程数")
    parser.add_argument("--gui", action="store_true", help="启动图形界面")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    
    args = parser.parse_args()
    
    if args.gui:
        # 启动GUI
        gui = BatchProcessorGUI()
        gui.run()
    elif args.directory:
        # 命令行模式
        processor = BatchProcessor(max_workers=args.workers, log_level=args.log_level)
        summary = processor.process_directory(
            args.directory,
            recursive=args.recursive,
            backup=args.backup
        )
        
        print("\n处理完成!")
        print(f"总文件数: {summary['summary']['total_files']}")
        print(f"成功处理: {summary['summary']['successful_files']}")
        print(f"失败文件: {summary['summary']['failed_files']}")
        print(f"成功率: {summary['summary']['success_rate']}")
        print(f"修复错误: {summary['summary']['total_errors_fixed']}")
        print(f"发现错误: {summary['summary']['total_errors_found']}")
        print(f"修复率: {summary['summary']['fix_rate']}")
        print(f"处理时间: {summary['summary']['processing_time']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 