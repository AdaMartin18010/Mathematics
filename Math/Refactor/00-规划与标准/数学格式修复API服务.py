#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复API服务
Math/Refactor项目数学格式修复RESTful API服务

作者: 数学知识体系重构项目组
时间: 2025年1月
版本: 1.0
"""

import os
import re
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask未安装，请运行: pip install flask flask-cors")

# 导入修复工具
import sys
sys.path.append(str(Path(__file__).parent))
from 数学格式修复执行脚本 import MathFormatFixer, MathFormatChecker
from 数学格式修复批量处理工具 import BatchProcessor
from 数学格式修复配置管理器 import ConfigManager

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """任务数据类"""
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class APIService:
    """API服务管理器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """初始化API服务"""
        if not FLASK_AVAILABLE:
            raise ImportError("Flask未安装，请先安装Flask")
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # 初始化Flask应用
        self.app = Flask(__name__)
        CORS(self.app)
        
        # 初始化工具
        self.fixer = MathFormatFixer("api")
        self.checker = MathFormatChecker("api")
        self.batch_processor = BatchProcessor(max_workers=4)
        self.config_manager = ConfigManager()
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('api_service.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 任务管理
        self.tasks: Dict[str, Task] = {}
        self.task_lock = threading.Lock()
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'services': {
                    'fixer': 'available',
                    'checker': 'available',
                    'batch_processor': 'available',
                    'config_manager': 'available'
                }
            })
        
        @self.app.route('/api/v1/check', methods=['POST'])
        def check_text():
            """检查文本API"""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({
                        'success': False,
                        'error': '缺少text参数'
                    }), 400
                
                text = data['text']
                options = data.get('options', {})
                
                # 检查语法错误
                errors = self.checker.check_math_syntax(text)
                
                # 统计错误类型
                error_stats = {
                    'syntax_errors': 0,
                    'symbol_errors': 0,
                    'format_errors': 0,
                    'other_errors': 0
                }
                
                for error in errors:
                    if 'syntax' in error['type'].lower():
                        error_stats['syntax_errors'] += 1
                    elif 'symbol' in error['type'].lower():
                        error_stats['symbol_errors'] += 1
                    elif 'format' in error['type'].lower():
                        error_stats['format_errors'] += 1
                    else:
                        error_stats['other_errors'] += 1
                
                return jsonify({
                    'success': True,
                    'data': {
                        'errors': errors,
                        'error_stats': error_stats,
                        'total_errors': len(errors),
                        'text_length': len(text)
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"检查文本时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/fix', methods=['POST'])
        def fix_text():
            """修复文本API"""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({
                        'success': False,
                        'error': '缺少text参数'
                    }), 400
                
                text = data['text']
                options = data.get('options', {})
                
                # 检查原始错误
                original_errors = self.checker.check_math_syntax(text)
                original_error_count = len(original_errors)
                
                # 修复文本
                fixed_text = text
                fixed_text = self.fixer.fix_bracket_mismatch(fixed_text)
                fixed_text = self.fixer.fix_greek_letters(fixed_text)
                fixed_text = self.fixer.fix_math_symbols(fixed_text)
                fixed_text = self.fixer.fix_inline_formulas(fixed_text)
                fixed_text = self.fixer.fix_block_formulas(fixed_text)
                fixed_text = self.fixer.fix_alignment(fixed_text)
                
                # 检查修复后的错误
                final_errors = self.checker.check_math_syntax(fixed_text)
                final_error_count = len(final_errors)
                
                # 计算修复的错误数量
                errors_fixed = original_error_count - final_error_count
                fix_rate = (errors_fixed / original_error_count * 100) if original_error_count > 0 else 0
                
                return jsonify({
                    'success': True,
                    'data': {
                        'original_text': text,
                        'fixed_text': fixed_text,
                        'original_errors': original_error_count,
                        'final_errors': final_error_count,
                        'errors_fixed': errors_fixed,
                        'fix_rate': f"{fix_rate:.1f}%",
                        'remaining_errors': final_errors
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"修复文本时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/tasks', methods=['POST'])
        def create_task():
            """创建任务API"""
            try:
                data = request.get_json()
                if not data or 'task_type' not in data:
                    return jsonify({
                        'success': False,
                        'error': '缺少task_type参数'
                    }), 400
                
                task_type = data['task_type']
                input_data = data.get('input_data', {})
                options = data.get('options', {})
                
                # 生成任务ID
                task_id = self._generate_task_id(task_type, input_data)
                
                # 创建任务
                task = Task(
                    task_id=task_id,
                    task_type=task_type,
                    status=TaskStatus.PENDING,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    input_data=input_data,
                    metadata=options
                )
                
                with self.task_lock:
                    self.tasks[task_id] = task
                
                # 启动任务处理
                if task_type == 'batch_fix':
                    self._start_batch_task(task)
                elif task_type == 'check_batch':
                    self._start_check_batch_task(task)
                else:
                    return jsonify({
                        'success': False,
                        'error': f'不支持的任务类型: {task_type}'
                    }), 400
                
                return jsonify({
                    'success': True,
                    'data': {
                        'task_id': task_id,
                        'status': task.status.value,
                        'created_at': task.created_at.isoformat()
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"创建任务时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/tasks/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            """获取任务状态API"""
            try:
                with self.task_lock:
                    task = self.tasks.get(task_id)
                
                if not task:
                    return jsonify({
                        'success': False,
                        'error': '任务不存在'
                    }), 404
                
                return jsonify({
                    'success': True,
                    'data': {
                        'task_id': task.task_id,
                        'task_type': task.task_type,
                        'status': task.status.value,
                        'progress': task.progress,
                        'created_at': task.created_at.isoformat(),
                        'updated_at': task.updated_at.isoformat(),
                        'error_message': task.error_message
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"获取任务状态时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/tasks/<task_id>/result', methods=['GET'])
        def get_task_result(task_id):
            """获取任务结果API"""
            try:
                with self.task_lock:
                    task = self.tasks.get(task_id)
                
                if not task:
                    return jsonify({
                        'success': False,
                        'error': '任务不存在'
                    }), 404
                
                if task.status != TaskStatus.COMPLETED:
                    return jsonify({
                        'success': False,
                        'error': f'任务尚未完成，当前状态: {task.status.value}'
                    }), 400
                
                return jsonify({
                    'success': True,
                    'data': {
                        'task_id': task.task_id,
                        'task_type': task.task_type,
                        'status': task.status.value,
                        'output_data': task.output_data,
                        'created_at': task.created_at.isoformat(),
                        'completed_at': task.updated_at.isoformat()
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"获取任务结果时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/tasks/<task_id>', methods=['DELETE'])
        def cancel_task(task_id):
            """取消任务API"""
            try:
                with self.task_lock:
                    task = self.tasks.get(task_id)
                
                if not task:
                    return jsonify({
                        'success': False,
                        'error': '任务不存在'
                    }), 404
                
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    return jsonify({
                        'success': False,
                        'error': f'无法取消已完成的任务，当前状态: {task.status.value}'
                    }), 400
                
                # 更新任务状态
                task.status = TaskStatus.CANCELLED
                task.updated_at = datetime.now()
                
                return jsonify({
                    'success': True,
                    'data': {
                        'task_id': task_id,
                        'status': task.status.value,
                        'message': '任务已取消'
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"取消任务时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/tasks', methods=['GET'])
        def list_tasks():
            """列出任务API"""
            try:
                page = int(request.args.get('page', 1))
                per_page = int(request.args.get('per_page', 10))
                status_filter = request.args.get('status')
                
                with self.task_lock:
                    tasks = list(self.tasks.values())
                
                # 过滤任务
                if status_filter:
                    tasks = [t for t in tasks if t.status.value == status_filter]
                
                # 排序（按创建时间倒序）
                tasks.sort(key=lambda x: x.created_at, reverse=True)
                
                # 分页
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                paginated_tasks = tasks[start_idx:end_idx]
                
                return jsonify({
                    'success': True,
                    'data': {
                        'tasks': [{
                            'task_id': t.task_id,
                            'task_type': t.task_type,
                            'status': t.status.value,
                            'progress': t.progress,
                            'created_at': t.created_at.isoformat(),
                            'updated_at': t.updated_at.isoformat()
                        } for t in paginated_tasks],
                        'pagination': {
                            'page': page,
                            'per_page': per_page,
                            'total': len(tasks),
                            'total_pages': (len(tasks) + per_page - 1) // per_page
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"列出任务时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/config', methods=['GET'])
        def get_config():
            """获取配置API"""
            try:
                config = self.config_manager.get_config()
                
                return jsonify({
                    'success': True,
                    'data': config,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"获取配置时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/config', methods=['PUT'])
        def update_config():
            """更新配置API"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({
                        'success': False,
                        'error': '缺少配置数据'
                    }), 400
                
                self.config_manager.update_config(data)
                
                return jsonify({
                    'success': True,
                    'data': {
                        'message': '配置已更新'
                    },
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"更新配置时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/v1/stats', methods=['GET'])
        def get_stats():
            """获取统计信息API"""
            try:
                with self.task_lock:
                    tasks = list(self.tasks.values())
                
                # 计算统计信息
                total_tasks = len(tasks)
                completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
                failed_tasks = len([t for t in tasks if t.status == TaskStatus.FAILED])
                processing_tasks = len([t for t in tasks if t.status == TaskStatus.PROCESSING])
                
                # 计算平均处理时间
                completed_tasks_with_time = [t for t in tasks if t.status == TaskStatus.COMPLETED and t.output_data]
                avg_processing_time = 0
                if completed_tasks_with_time:
                    total_time = sum((t.updated_at - t.created_at).total_seconds() for t in completed_tasks_with_time)
                    avg_processing_time = total_time / len(completed_tasks_with_time)
                
                stats = {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'failed_tasks': failed_tasks,
                    'processing_tasks': processing_tasks,
                    'success_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                    'avg_processing_time': f"{avg_processing_time:.2f}s"
                }
                
                return jsonify({
                    'success': True,
                    'data': stats,
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                self.logger.error(f"获取统计信息时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
    
    def _generate_task_id(self, task_type: str, input_data: Dict[str, Any]) -> str:
        """生成任务ID"""
        content = f"{task_type}_{json.dumps(input_data, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _start_batch_task(self, task: Task):
        """启动批量修复任务"""
        def process_batch():
            try:
                with self.task_lock:
                    task.status = TaskStatus.PROCESSING
                    task.updated_at = datetime.now()
                
                # 获取文件列表
                files = task.input_data.get('files', [])
                if not files:
                    raise ValueError("没有提供文件列表")
                
                # 处理文件
                results = []
                for i, file_path in enumerate(files):
                    try:
                        # 检查任务是否被取消
                        with self.task_lock:
                            if task.status == TaskStatus.CANCELLED:
                                return
                        
                        # 读取文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 检查原始错误
                        original_errors = self.checker.check_math_syntax(content)
                        original_error_count = len(original_errors)
                        
                        # 修复内容
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
                        
                        # 保存修复后的文件
                        fixed_file_path = Path(file_path).parent / f"fixed_{Path(file_path).name}"
                        with open(fixed_file_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        
                        results.append({
                            'file': file_path,
                            'status': 'success',
                            'original_errors': original_error_count,
                            'final_errors': final_error_count,
                            'errors_fixed': original_error_count - final_error_count,
                            'fixed_file': str(fixed_file_path)
                        })
                        
                    except Exception as e:
                        results.append({
                            'file': file_path,
                            'status': 'failed',
                            'error': str(e),
                            'errors_fixed': 0
                        })
                    
                    # 更新进度
                    progress = ((i + 1) / len(files)) * 100
                    with self.task_lock:
                        task.progress = progress
                        task.updated_at = datetime.now()
                
                # 计算总体统计
                total_errors_found = sum(r.get('original_errors', 0) for r in results)
                total_errors_fixed = sum(r.get('errors_fixed', 0) for r in results)
                successful_files = len([r for r in results if r['status'] == 'success'])
                
                # 更新任务结果
                with self.task_lock:
                    task.output_data = {
                        'total_files': len(files),
                        'successful_files': successful_files,
                        'failed_files': len(files) - successful_files,
                        'total_errors_found': total_errors_found,
                        'total_errors_fixed': total_errors_fixed,
                        'fix_rate': f"{(total_errors_fixed / total_errors_found * 100) if total_errors_found > 0 else 0:.1f}%",
                        'results': results
                    }
                    task.status = TaskStatus.COMPLETED
                    task.updated_at = datetime.now()
                
            except Exception as e:
                with self.task_lock:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.updated_at = datetime.now()
        
        # 启动处理线程
        thread = threading.Thread(target=process_batch)
        thread.daemon = True
        thread.start()
    
    def _start_check_batch_task(self, task: Task):
        """启动批量检查任务"""
        def process_check_batch():
            try:
                with self.task_lock:
                    task.status = TaskStatus.PROCESSING
                    task.updated_at = datetime.now()
                
                # 获取文件列表
                files = task.input_data.get('files', [])
                if not files:
                    raise ValueError("没有提供文件列表")
                
                # 处理文件
                results = []
                for i, file_path in enumerate(files):
                    try:
                        # 检查任务是否被取消
                        with self.task_lock:
                            if task.status == TaskStatus.CANCELLED:
                                return
                        
                        # 读取文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 检查错误
                        errors = self.checker.check_math_syntax(content)
                        error_count = len(errors)
                        
                        results.append({
                            'file': file_path,
                            'status': 'success',
                            'error_count': error_count,
                            'errors': errors
                        })
                        
                    except Exception as e:
                        results.append({
                            'file': file_path,
                            'status': 'failed',
                            'error': str(e),
                            'error_count': 0
                        })
                    
                    # 更新进度
                    progress = ((i + 1) / len(files)) * 100
                    with self.task_lock:
                        task.progress = progress
                        task.updated_at = datetime.now()
                
                # 计算总体统计
                total_errors = sum(r.get('error_count', 0) for r in results)
                successful_files = len([r for r in results if r['status'] == 'success'])
                
                # 更新任务结果
                with self.task_lock:
                    task.output_data = {
                        'total_files': len(files),
                        'successful_files': successful_files,
                        'failed_files': len(files) - successful_files,
                        'total_errors': total_errors,
                        'results': results
                    }
                    task.status = TaskStatus.COMPLETED
                    task.updated_at = datetime.now()
                
            except Exception as e:
                with self.task_lock:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.updated_at = datetime.now()
        
        # 启动处理线程
        thread = threading.Thread(target=process_check_batch)
        thread.daemon = True
        thread.start()
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """清理旧任务"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.task_lock:
            old_tasks = [
                task_id for task_id, task in self.tasks.items()
                if task.updated_at < cutoff_time and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            ]
            
            for task_id in old_tasks:
                del self.tasks[task_id]
        
        if old_tasks:
            self.logger.info(f"清理了 {len(old_tasks)} 个旧任务")
    
    def run(self):
        """运行API服务"""
        # 启动清理任务
        def cleanup_worker():
            while True:
                time.sleep(3600)  # 每小时清理一次
                self.cleanup_old_tasks()
        
        cleanup_thread = threading.Thread(target=cleanup_worker)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        # 启动Flask应用
        self.logger.info(f"启动API服务: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数学格式修复API服务")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=5000, help="端口号")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    try:
        # 检查Flask是否可用
        if not FLASK_AVAILABLE:
            print("错误: Flask未安装")
            print("请运行以下命令安装Flask:")
            print("pip install flask flask-cors")
            return
        
        # 创建并运行API服务
        api_service = APIService(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        api_service.run()
        
    except Exception as e:
        print(f"启动API服务时发生错误: {e}")

if __name__ == "__main__":
    main() 