#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复Web界面
Math/Refactor项目数学格式修复Web界面

作者: 数学知识体系重构项目组
时间: 2025年1月
版本: 1.0
"""

import os
import re
import json
import time
import base64
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

try:
    from flask import Flask, render_template, request, jsonify, send_file, session
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

class WebInterface:
    """Web界面管理器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
        """初始化Web界面"""
        if not FLASK_AVAILABLE:
            raise ImportError("Flask未安装，请先安装Flask")
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # 初始化Flask应用
        self.app = Flask(__name__)
        self.app.secret_key = 'math_format_fixer_secret_key'
        CORS(self.app)
        
        # 初始化工具
        self.fixer = MathFormatFixer("web")
        self.checker = MathFormatChecker("web")
        self.batch_processor = BatchProcessor(max_workers=4)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('web_interface.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 处理状态
        self.processing_status = {}
        self.processing_results = {}
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """注册路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('index.html')
        
        @self.app.route('/api/check', methods=['POST'])
        def check_text():
            """检查文本"""
            try:
                data = request.get_json()
                text = data.get('text', '')
                
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
                    'errors': errors,
                    'error_stats': error_stats,
                    'total_errors': len(errors)
                })
            
            except Exception as e:
                self.logger.error(f"检查文本时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/fix', methods=['POST'])
        def fix_text():
            """修复文本"""
            try:
                data = request.get_json()
                text = data.get('text', '')
                
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
                    'original_text': text,
                    'fixed_text': fixed_text,
                    'original_errors': original_error_count,
                    'final_errors': final_error_count,
                    'errors_fixed': errors_fixed,
                    'fix_rate': f"{fix_rate:.1f}%",
                    'remaining_errors': final_errors
                })
            
            except Exception as e:
                self.logger.error(f"修复文本时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/batch/upload', methods=['POST'])
        def upload_files():
            """上传文件进行批量处理"""
            try:
                if 'files' not in request.files:
                    return jsonify({
                        'success': False,
                        'error': '没有上传文件'
                    }), 400
                
                files = request.files.getlist('files')
                if not files or files[0].filename == '':
                    return jsonify({
                        'success': False,
                        'error': '没有选择文件'
                    }), 400
                
                # 创建临时目录
                temp_dir = Path('temp_uploads') / f"batch_{int(time.time())}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存上传的文件
                uploaded_files = []
                for file in files:
                    if file and file.filename:
                        filename = secure_filename(file.filename)
                        file_path = temp_dir / filename
                        file.save(file_path)
                        uploaded_files.append(str(file_path))
                
                # 启动批量处理
                session['batch_id'] = str(int(time.time()))
                session['batch_files'] = uploaded_files
                
                # 在新线程中处理
                thread = threading.Thread(
                    target=self._process_batch_files,
                    args=(uploaded_files, session['batch_id'])
                )
                thread.daemon = True
                thread.start()
                
                return jsonify({
                    'success': True,
                    'batch_id': session['batch_id'],
                    'files_count': len(uploaded_files),
                    'message': '批量处理已开始'
                })
            
            except Exception as e:
                self.logger.error(f"上传文件时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/batch/status/<batch_id>')
        def get_batch_status(batch_id):
            """获取批量处理状态"""
            try:
                if batch_id in self.processing_status:
                    status = self.processing_status[batch_id]
                    return jsonify({
                        'success': True,
                        'status': status
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': '批处理ID不存在'
                    }), 404
            
            except Exception as e:
                self.logger.error(f"获取批处理状态时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/batch/result/<batch_id>')
        def get_batch_result(batch_id):
            """获取批量处理结果"""
            try:
                if batch_id in self.processing_results:
                    result = self.processing_results[batch_id]
                    return jsonify({
                        'success': True,
                        'result': result
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': '批处理结果不存在'
                    }), 404
            
            except Exception as e:
                self.logger.error(f"获取批处理结果时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/download/<batch_id>')
        def download_batch_result(batch_id):
            """下载批量处理结果"""
            try:
                if batch_id in self.processing_results:
                    result = self.processing_results[batch_id]
                    
                    # 创建结果文件
                    result_file = Path(f"batch_result_{batch_id}.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    return send_file(
                        result_file,
                        as_attachment=True,
                        download_name=f"batch_result_{batch_id}.json"
                    )
                else:
                    return jsonify({
                        'success': False,
                        'error': '批处理结果不存在'
                    }), 404
            
            except Exception as e:
                self.logger.error(f"下载批处理结果时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/config')
        def get_config():
            """获取配置信息"""
            try:
                # 这里可以返回当前配置信息
                config = {
                    'fix_rules': [
                        {'name': 'bracket_mismatch', 'enabled': True},
                        {'name': 'greek_letters', 'enabled': True},
                        {'name': 'math_symbols', 'enabled': True}
                    ],
                    'check_rules': [
                        {'name': 'syntax_check', 'enabled': True},
                        {'name': 'symbol_check', 'enabled': True}
                    ],
                    'processing_config': {
                        'max_workers': 4,
                        'backup_enabled': True
                    }
                }
                
                return jsonify({
                    'success': True,
                    'config': config
                })
            
            except Exception as e:
                self.logger.error(f"获取配置时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/stats')
        def get_stats():
            """获取统计信息"""
            try:
                stats = {
                    'total_processed': len(self.processing_results),
                    'active_batches': len([s for s in self.processing_status.values() if s['status'] == 'processing']),
                    'completed_batches': len([s for s in self.processing_status.values() if s['status'] == 'completed']),
                    'failed_batches': len([s for s in self.processing_status.values() if s['status'] == 'failed'])
                }
                
                return jsonify({
                    'success': True,
                    'stats': stats
                })
            
            except Exception as e:
                self.logger.error(f"获取统计信息时发生错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def _process_batch_files(self, files: List[str], batch_id: str):
        """处理批量文件"""
        try:
            # 更新状态
            self.processing_status[batch_id] = {
                'status': 'processing',
                'progress': 0,
                'total_files': len(files),
                'processed_files': 0,
                'start_time': datetime.now().isoformat()
            }
            
            # 处理文件
            results = []
            for i, file_path in enumerate(files):
                try:
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
                self.processing_status[batch_id]['processed_files'] = i + 1
                self.processing_status[batch_id]['progress'] = ((i + 1) / len(files)) * 100
            
            # 计算总体统计
            total_errors_found = sum(r.get('original_errors', 0) for r in results)
            total_errors_fixed = sum(r.get('errors_fixed', 0) for r in results)
            successful_files = len([r for r in results if r['status'] == 'success'])
            
            # 保存结果
            self.processing_results[batch_id] = {
                'batch_id': batch_id,
                'total_files': len(files),
                'successful_files': successful_files,
                'failed_files': len(files) - successful_files,
                'total_errors_found': total_errors_found,
                'total_errors_fixed': total_errors_fixed,
                'fix_rate': f"{(total_errors_fixed / total_errors_found * 100) if total_errors_found > 0 else 0:.1f}%",
                'results': results,
                'end_time': datetime.now().isoformat()
            }
            
            # 更新状态
            self.processing_status[batch_id]['status'] = 'completed'
            self.processing_status[batch_id]['progress'] = 100
            
        except Exception as e:
            self.logger.error(f"批量处理时发生错误: {e}")
            self.processing_status[batch_id]['status'] = 'failed'
            self.processing_status[batch_id]['error'] = str(e)
    
    def create_templates(self):
        """创建HTML模板"""
        templates_dir = Path('templates')
        templates_dir.mkdir(exist_ok=True)
        
        # 创建主页模板
        index_html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数学格式修复工具</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .content {
            padding: 30px;
        }
        .tabs {
            display: flex;
            border-bottom: 2px solid #eee;
            margin-bottom: 30px;
        }
        .tab {
            padding: 15px 30px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
        }
        .tab:hover {
            background-color: #f8f9fa;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #333;
        }
        textarea, input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s ease;
            box-sizing: border-box;
        }
        textarea:focus, input[type="file"]:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            min-height: 200px;
            resize: vertical;
            font-family: 'Courier New', monospace;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.3s ease;
            margin-right: 10px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
            background-color: #f8f9fa;
        }
        .error {
            border-left-color: #dc3545;
            background-color: #f8d7da;
        }
        .success {
            border-left-color: #28a745;
            background-color: #d4edda;
        }
        .progress {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .file-list {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 10px;
        }
        .file-item {
            padding: 8px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .file-item:last-child {
            border-bottom: none;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .status-success {
            background-color: #d4edda;
            color: #155724;
        }
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>数学格式修复工具</h1>
            <p>专业的数学文档格式修复和标准化工具</p>
        </div>
        
        <div class="content">
            <div class="tabs">
                <div class="tab active" onclick="switchTab('single')">单文件处理</div>
                <div class="tab" onclick="switchTab('batch')">批量处理</div>
                <div class="tab" onclick="switchTab('stats')">统计信息</div>
            </div>
            
            <!-- 单文件处理 -->
            <div id="single" class="tab-content active">
                <div class="form-group">
                    <label for="input-text">输入数学文本：</label>
                    <textarea id="input-text" placeholder="在此输入包含数学公式的文本..."></textarea>
                </div>
                
                <div class="form-group">
                    <button class="btn" onclick="checkText()">检查错误</button>
                    <button class="btn" onclick="fixText()">修复格式</button>
                    <button class="btn" onclick="clearText()">清空</button>
                </div>
                
                <div id="single-result"></div>
            </div>
            
            <!-- 批量处理 -->
            <div id="batch" class="tab-content">
                <div class="form-group">
                    <label for="batch-files">选择文件：</label>
                    <input type="file" id="batch-files" multiple accept=".md,.txt">
                </div>
                
                <div class="form-group">
                    <button class="btn" onclick="uploadFiles()">开始批量处理</button>
                    <button class="btn" onclick="clearBatch()">清空</button>
                </div>
                
                <div id="batch-progress" style="display: none;">
                    <h3>处理进度</h3>
                    <div class="progress">
                        <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
                    </div>
                    <p id="progress-text">准备中...</p>
                </div>
                
                <div id="batch-result"></div>
            </div>
            
            <!-- 统计信息 -->
            <div id="stats" class="tab-content">
                <div class="stats" id="stats-content">
                    <div class="stat-card">
                        <div class="stat-number" id="total-processed">0</div>
                        <div class="stat-label">总处理数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="active-batches">0</div>
                        <div class="stat-label">活跃批处理</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="completed-batches">0</div>
                        <div class="stat-label">已完成</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="failed-batches">0</div>
                        <div class="stat-label">失败数</div>
                    </div>
                </div>
                
                <div class="form-group">
                    <button class="btn" onclick="loadStats()">刷新统计</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentBatchId = null;
        let progressInterval = null;

        function switchTab(tabName) {
            // 隐藏所有标签内容
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // 移除所有标签的active类
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // 显示选中的标签内容
            document.getElementById(tabName).classList.add('active');
            
            // 添加active类到选中的标签
            event.target.classList.add('active');
        }

        async function checkText() {
            const text = document.getElementById('input-text').value;
            if (!text.trim()) {
                alert('请输入要检查的文本');
                return;
            }

            try {
                const response = await fetch('/api/check', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });

                const result = await response.json();
                
                if (result.success) {
                    displayCheckResult(result);
                } else {
                    displayError(result.error);
                }
            } catch (error) {
                displayError('检查过程中发生错误: ' + error.message);
            }
        }

        async function fixText() {
            const text = document.getElementById('input-text').value;
            if (!text.trim()) {
                alert('请输入要修复的文本');
                return;
            }

            try {
                const response = await fetch('/api/fix', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });

                const result = await response.json();
                
                if (result.success) {
                    displayFixResult(result);
                } else {
                    displayError(result.error);
                }
            } catch (error) {
                displayError('修复过程中发生错误: ' + error.message);
            }
        }

        function displayCheckResult(result) {
            const resultDiv = document.getElementById('single-result');
            resultDiv.innerHTML = `
                <div class="result">
                    <h3>检查结果</h3>
                    <p><strong>发现错误:</strong> ${result.total_errors} 个</p>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-number">${result.error_stats.syntax_errors}</div>
                            <div class="stat-label">语法错误</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${result.error_stats.symbol_errors}</div>
                            <div class="stat-label">符号错误</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${result.error_stats.format_errors}</div>
                            <div class="stat-label">格式错误</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${result.error_stats.other_errors}</div>
                            <div class="stat-label">其他错误</div>
                        </div>
                    </div>
                    ${result.errors.length > 0 ? `
                        <h4>详细错误信息:</h4>
                        <div style="max-height: 300px; overflow-y: auto;">
                            ${result.errors.map(error => `
                                <div style="padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 4px;">
                                    <strong>${error.type}:</strong> ${error.message}
                                    ${error.suggestions ? `<br><small>建议: ${error.suggestions.join(', ')}</small>` : ''}
                                </div>
                            `).join('')}
                        </div>
                    ` : '<p>未发现错误！</p>'}
                </div>
            `;
        }

        function displayFixResult(result) {
            const resultDiv = document.getElementById('single-result');
            resultDiv.innerHTML = `
                <div class="result success">
                    <h3>修复结果</h3>
                    <p><strong>修复率:</strong> ${result.fix_rate}</p>
                    <p><strong>修复错误:</strong> ${result.errors_fixed} 个</p>
                    <p><strong>剩余错误:</strong> ${result.final_errors} 个</p>
                    
                    <h4>修复后的文本:</h4>
                    <textarea style="width: 100%; min-height: 150px; margin-top: 10px;" readonly>${result.fixed_text}</textarea>
                    
                    ${result.remaining_errors.length > 0 ? `
                        <h4>剩余错误:</h4>
                        <div style="max-height: 200px; overflow-y: auto;">
                            ${result.remaining_errors.map(error => `
                                <div style="padding: 8px; margin: 3px 0; background: #fff3cd; border-radius: 4px; font-size: 12px;">
                                    <strong>${error.type}:</strong> ${error.message}
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        }

        function displayError(message) {
            const resultDiv = document.getElementById('single-result');
            resultDiv.innerHTML = `
                <div class="result error">
                    <h3>错误</h3>
                    <p>${message}</p>
                </div>
            `;
        }

        function clearText() {
            document.getElementById('input-text').value = '';
            document.getElementById('single-result').innerHTML = '';
        }

        async function uploadFiles() {
            const fileInput = document.getElementById('batch-files');
            const files = fileInput.files;
            
            if (files.length === 0) {
                alert('请选择要处理的文件');
                return;
            }

            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }

            try {
                const response = await fetch('/api/batch/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.success) {
                    currentBatchId = result.batch_id;
                    startProgressMonitoring();
                    displayBatchProgress(result);
                } else {
                    displayError(result.error);
                }
            } catch (error) {
                displayError('上传文件时发生错误: ' + error.message);
            }
        }

        function startProgressMonitoring() {
            if (progressInterval) {
                clearInterval(progressInterval);
            }
            
            progressInterval = setInterval(async () => {
                if (!currentBatchId) return;
                
                try {
                    const response = await fetch(`/api/batch/status/${currentBatchId}`);
                    const result = await response.json();
                    
                    if (result.success) {
                        updateProgress(result.status);
                        
                        if (result.status.status === 'completed' || result.status.status === 'failed') {
                            clearInterval(progressInterval);
                            loadBatchResult();
                        }
                    }
                } catch (error) {
                    console.error('获取进度时发生错误:', error);
                }
            }, 1000);
        }

        function updateProgress(status) {
            const progressBar = document.getElementById('progress-bar');
            const progressText = document.getElementById('progress-text');
            
            progressBar.style.width = status.progress + '%';
            progressText.textContent = `处理中... ${status.processed_files}/${status.total_files} (${status.progress.toFixed(1)}%)`;
        }

        async function loadBatchResult() {
            if (!currentBatchId) return;
            
            try {
                const response = await fetch(`/api/batch/result/${currentBatchId}`);
                const result = await response.json();
                
                if (result.success) {
                    displayBatchResult(result.result);
                } else {
                    displayError(result.error);
                }
            } catch (error) {
                displayError('获取批处理结果时发生错误: ' + error.message);
            }
        }

        function displayBatchResult(result) {
            const resultDiv = document.getElementById('batch-result');
            resultDiv.innerHTML = `
                <div class="result success">
                    <h3>批量处理完成</h3>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-number">${result.total_files}</div>
                            <div class="stat-label">总文件数</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${result.successful_files}</div>
                            <div class="stat-label">成功处理</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${result.total_errors_fixed}</div>
                            <div class="stat-label">修复错误</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${result.fix_rate}</div>
                            <div class="stat-label">修复率</div>
                        </div>
                    </div>
                    
                    <h4>处理详情:</h4>
                    <div class="file-list">
                        ${result.results.map(file => `
                            <div class="file-item">
                                <span>${file.file.split('/').pop()}</span>
                                <span class="status-badge ${file.status === 'success' ? 'status-success' : 'status-error'}">
                                    ${file.status === 'success' ? '成功' : '失败'}
                                </span>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <button class="btn" onclick="downloadResult('${currentBatchId}')">下载结果</button>
                    </div>
                </div>
            `;
        }

        function displayBatchProgress(result) {
            const progressDiv = document.getElementById('batch-progress');
            progressDiv.style.display = 'block';
            
            const resultDiv = document.getElementById('batch-result');
            resultDiv.innerHTML = `
                <div class="result">
                    <h3>批量处理已开始</h3>
                    <p>批处理ID: ${result.batch_id}</p>
                    <p>文件数量: ${result.files_count}</p>
                    <div class="loading"></div>
                    <p>正在处理中...</p>
                </div>
            `;
        }

        function clearBatch() {
            document.getElementById('batch-files').value = '';
            document.getElementById('batch-progress').style.display = 'none';
            document.getElementById('batch-result').innerHTML = '';
            if (progressInterval) {
                clearInterval(progressInterval);
            }
            currentBatchId = null;
        }

        async function downloadResult(batchId) {
            try {
                window.open(`/api/download/${batchId}`, '_blank');
            } catch (error) {
                displayError('下载结果时发生错误: ' + error.message);
            }
        }

        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('total-processed').textContent = result.stats.total_processed;
                    document.getElementById('active-batches').textContent = result.stats.active_batches;
                    document.getElementById('completed-batches').textContent = result.stats.completed_batches;
                    document.getElementById('failed-batches').textContent = result.stats.failed_batches;
                }
            } catch (error) {
                console.error('加载统计信息时发生错误:', error);
            }
        }

        // 页面加载时加载统计信息
        window.onload = function() {
            loadStats();
        };
    </script>
</body>
</html>'''
        
        with open(templates_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        self.logger.info("HTML模板已创建")
    
    def run(self):
        """运行Web界面"""
        # 创建模板
        self.create_templates()
        
        # 启动Flask应用
        self.logger.info(f"启动Web界面: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数学格式修复Web界面")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8080, help="端口号")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    try:
        # 检查Flask是否可用
        if not FLASK_AVAILABLE:
            print("错误: Flask未安装")
            print("请运行以下命令安装Flask:")
            print("pip install flask flask-cors")
            return
        
        # 创建并运行Web界面
        web_interface = WebInterface(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        web_interface.run()
        
    except Exception as e:
        print(f"启动Web界面时发生错误: {e}")

if __name__ == "__main__":
    main() 