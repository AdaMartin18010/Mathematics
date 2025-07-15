#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复监控仪表板
提供实时监控、数据可视化和告警功能
"""

import os
import sys
import json
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import psutil
import requests
from flask import Flask, render_template, jsonify, request
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from collections import deque
import sqlite3

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_sent: int
    network_recv: int
    active_processes: int

@dataclass
class ServiceMetrics:
    """服务指标"""
    service_name: str
    status: str  # 'running', 'stopped', 'error'
    response_time: float
    error_count: int
    request_count: int
    last_check: datetime

@dataclass
class Alert:
    """告警"""
    id: str
    level: str  # 'info', 'warning', 'error', 'critical'
    message: str
    timestamp: datetime
    service: str
    resolved: bool

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics = deque(maxlen=max_history)
        self.service_metrics = {}
        self.alerts = []
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('MetricsCollector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # 网络使用情况
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent
            network_recv = network.bytes_recv
            
            # 活跃进程数
            active_processes = len(psutil.pids())
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_sent=network_sent,
                network_recv=network_recv,
                active_processes=active_processes
            )
            
            self.system_metrics.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集系统指标失败: {str(e)}")
            return None
    
    def collect_service_metrics(self, service_url: str, service_name: str) -> ServiceMetrics:
        """收集服务指标"""
        try:
            start_time = time.time()
            
            # 检查服务状态
            response = requests.get(f"{service_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            if response.status_code == 200:
                status = 'running'
                error_count = 0
            else:
                status = 'error'
                error_count = 1
            
            # 获取请求统计
            try:
                stats_response = requests.get(f"{service_url}/stats", timeout=5)
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    request_count = stats.get('total_requests', 0)
                else:
                    request_count = 0
            except:
                request_count = 0
            
            metrics = ServiceMetrics(
                service_name=service_name,
                status=status,
                response_time=response_time,
                error_count=error_count,
                request_count=request_count,
                last_check=datetime.now()
            )
            
            self.service_metrics[service_name] = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"收集服务 {service_name} 指标失败: {str(e)}")
            
            # 创建错误指标
            error_metrics = ServiceMetrics(
                service_name=service_name,
                status='error',
                response_time=0,
                error_count=1,
                request_count=0,
                last_check=datetime.now()
            )
            
            self.service_metrics[service_name] = error_metrics
            return error_metrics
    
    def check_alerts(self) -> List[Alert]:
        """检查告警"""
        new_alerts = []
        
        # 检查系统指标告警
        if self.system_metrics:
            latest_metrics = self.system_metrics[-1]
            
            # CPU使用率告警
            if latest_metrics.cpu_percent > 90:
                alert = Alert(
                    id=f"cpu_high_{int(time.time())}",
                    level='warning',
                    message=f"CPU使用率过高: {latest_metrics.cpu_percent:.1f}%",
                    timestamp=datetime.now(),
                    service='system',
                    resolved=False
                )
                new_alerts.append(alert)
            
            # 内存使用率告警
            if latest_metrics.memory_percent > 85:
                alert = Alert(
                    id=f"memory_high_{int(time.time())}",
                    level='warning',
                    message=f"内存使用率过高: {latest_metrics.memory_percent:.1f}%",
                    timestamp=datetime.now(),
                    service='system',
                    resolved=False
                )
                new_alerts.append(alert)
            
            # 磁盘使用率告警
            if latest_metrics.disk_percent > 90:
                alert = Alert(
                    id=f"disk_high_{int(time.time())}",
                    level='critical',
                    message=f"磁盘使用率过高: {latest_metrics.disk_percent:.1f}%",
                    timestamp=datetime.now(),
                    service='system',
                    resolved=False
                )
                new_alerts.append(alert)
        
        # 检查服务指标告警
        for service_name, metrics in self.service_metrics.items():
            # 服务停止告警
            if metrics.status == 'error':
                alert = Alert(
                    id=f"service_error_{service_name}_{int(time.time())}",
                    level='error',
                    message=f"服务 {service_name} 无响应",
                    timestamp=datetime.now(),
                    service=service_name,
                    resolved=False
                )
                new_alerts.append(alert)
            
            # 响应时间告警
            elif metrics.response_time > 5000:  # 5秒
                alert = Alert(
                    id=f"response_slow_{service_name}_{int(time.time())}",
                    level='warning',
                    message=f"服务 {service_name} 响应时间过长: {metrics.response_time:.0f}ms",
                    timestamp=datetime.now(),
                    service=service_name,
                    resolved=False
                )
                new_alerts.append(alert)
        
        # 添加新告警
        self.alerts.extend(new_alerts)
        
        # 清理已解决的告警（超过1小时）
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
        
        return new_alerts

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_file: str = 'monitoring.db'):
        self.db_file = db_file
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # 创建系统指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                network_sent INTEGER,
                network_recv INTEGER,
                active_processes INTEGER
            )
        ''')
        
        # 创建服务指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS service_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_name TEXT NOT NULL,
                status TEXT,
                response_time REAL,
                error_count INTEGER,
                request_count INTEGER,
                last_check TEXT
            )
        ''')
        
        # 创建告警表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                level TEXT,
                message TEXT,
                timestamp TEXT,
                service TEXT,
                resolved BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_system_metrics(self, metrics: SystemMetrics):
        """保存系统指标"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_metrics 
            (timestamp, cpu_percent, memory_percent, disk_percent, 
             network_sent, network_recv, active_processes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.disk_percent,
            metrics.network_sent,
            metrics.network_recv,
            metrics.active_processes
        ))
        
        conn.commit()
        conn.close()
    
    def save_service_metrics(self, metrics: ServiceMetrics):
        """保存服务指标"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # 更新或插入服务指标
        cursor.execute('''
            INSERT OR REPLACE INTO service_metrics 
            (service_name, status, response_time, error_count, request_count, last_check)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metrics.service_name,
            metrics.status,
            metrics.response_time,
            metrics.error_count,
            metrics.request_count,
            metrics.last_check.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def save_alert(self, alert: Alert):
        """保存告警"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO alerts 
            (id, level, message, timestamp, service, resolved)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            alert.id,
            alert.level,
            alert.message,
            alert.timestamp.isoformat(),
            alert.service,
            alert.resolved
        ))
        
        conn.commit()
        conn.close()
    
    def get_system_metrics_history(self, hours: int = 24) -> List[Dict]:
        """获取系统指标历史"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT * FROM system_metrics 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (cutoff_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': row[1],
                'cpu_percent': row[2],
                'memory_percent': row[3],
                'disk_percent': row[4],
                'network_sent': row[5],
                'network_recv': row[6],
                'active_processes': row[7]
            }
            for row in results
        ]
    
    def get_service_metrics(self) -> List[Dict]:
        """获取服务指标"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM service_metrics')
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'service_name': row[1],
                'status': row[2],
                'response_time': row[3],
                'error_count': row[4],
                'request_count': row[5],
                'last_check': row[6]
            }
            for row in results
        ]
    
    def get_alerts(self, resolved: bool = None) -> List[Dict]:
        """获取告警"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        if resolved is not None:
            cursor.execute('SELECT * FROM alerts WHERE resolved = ? ORDER BY timestamp DESC', (resolved,))
        else:
            cursor.execute('SELECT * FROM alerts ORDER BY timestamp DESC')
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'level': row[1],
                'message': row[2],
                'timestamp': row[3],
                'service': row[4],
                'resolved': bool(row[5])
            }
            for row in results
        ]

class MonitoringDashboard:
    """监控仪表板"""
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.db_manager = DatabaseManager()
        self.app = Flask(__name__)
        self.setup_routes()
        self.monitoring_thread = None
        self.stop_monitoring = False
        
    def setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def dashboard():
            return self._render_dashboard()
        
        @self.app.route('/api/metrics/system')
        def api_system_metrics():
            return jsonify(self._get_system_metrics())
        
        @self.app.route('/api/metrics/services')
        def api_service_metrics():
            return jsonify(self._get_service_metrics())
        
        @self.app.route('/api/alerts')
        def api_alerts():
            resolved = request.args.get('resolved', 'false').lower() == 'true'
            return jsonify(self._get_alerts(resolved))
        
        @self.app.route('/api/charts/system')
        def api_system_charts():
            return jsonify(self._generate_system_charts())
        
        @self.app.route('/api/charts/services')
        def api_service_charts():
            return jsonify(self._generate_service_charts())
    
    def _render_dashboard(self) -> str:
        """渲染仪表板HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>数学格式修复监控仪表板</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333; }
        .metric-value { font-size: 24px; font-weight: bold; color: #667eea; }
        .metric-unit { font-size: 14px; color: #666; }
        .status-running { color: #28a745; }
        .status-error { color: #dc3545; }
        .status-warning { color: #ffc107; }
        .alerts-section { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
        .alert-error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .alert-warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .alert-info { background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
        .charts-section { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .refresh-btn { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-bottom: 20px; }
        .refresh-btn:hover { background: #5a6fd8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>数学格式修复监控仪表板</h1>
            <p>实时系统监控和服务状态</p>
        </div>
        
        <button class="refresh-btn" onclick="refreshData()">刷新数据</button>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">CPU使用率</div>
                <div class="metric-value" id="cpu-percent">--</div>
                <div class="metric-unit">%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">内存使用率</div>
                <div class="metric-value" id="memory-percent">--</div>
                <div class="metric-unit">%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">磁盘使用率</div>
                <div class="metric-value" id="disk-percent">--</div>
                <div class="metric-unit">%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">活跃进程</div>
                <div class="metric-value" id="active-processes">--</div>
                <div class="metric-unit">个</div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Web界面</div>
                <div class="metric-value" id="web-status">--</div>
                <div class="metric-unit" id="web-response">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">API服务</div>
                <div class="metric-value" id="api-status">--</div>
                <div class="metric-unit" id="api-response">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">批量处理器</div>
                <div class="metric-value" id="batch-status">--</div>
                <div class="metric-unit" id="batch-response">--</div>
            </div>
        </div>
        
        <div class="alerts-section">
            <h2>告警信息</h2>
            <div id="alerts-container"></div>
        </div>
        
        <div class="charts-section">
            <div class="chart-container">
                <h3>系统资源使用趋势</h3>
                <canvas id="system-chart"></canvas>
            </div>
            <div class="chart-container">
                <h3>服务响应时间</h3>
                <canvas id="service-chart"></canvas>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        let systemChart, serviceChart;
        
        function refreshData() {
            // 刷新系统指标
            fetch('/api/metrics/system')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('cpu-percent').textContent = data.cpu_percent?.toFixed(1) || '--';
                    document.getElementById('memory-percent').textContent = data.memory_percent?.toFixed(1) || '--';
                    document.getElementById('disk-percent').textContent = data.disk_percent?.toFixed(1) || '--';
                    document.getElementById('active-processes').textContent = data.active_processes || '--';
                });
            
            // 刷新服务指标
            fetch('/api/metrics/services')
                .then(response => response.json())
                .then(data => {
                    data.forEach(service => {
                        const statusElement = document.getElementById(service.service_name + '-status');
                        const responseElement = document.getElementById(service.service_name + '-response');
                        
                        if (statusElement && responseElement) {
                            statusElement.textContent = service.status;
                            statusElement.className = 'metric-value status-' + service.status;
                            responseElement.textContent = service.response_time?.toFixed(0) + 'ms' || '--';
                        }
                    });
                });
            
            // 刷新告警
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('alerts-container');
                    container.innerHTML = '';
                    
                    data.forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = 'alert alert-' + alert.level;
                        alertDiv.innerHTML = `
                            <strong>${alert.level.toUpperCase()}</strong>: ${alert.message}
                            <br><small>${alert.timestamp} - ${alert.service}</small>
                        `;
                        container.appendChild(alertDiv);
                    });
                });
            
            // 刷新图表
            updateCharts();
        }
        
        function updateCharts() {
            // 系统图表
            fetch('/api/charts/system')
                .then(response => response.json())
                .then(data => {
                    if (systemChart) {
                        systemChart.destroy();
                    }
                    
                    const ctx = document.getElementById('system-chart').getContext('2d');
                    systemChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.labels,
                            datasets: [
                                {
                                    label: 'CPU使用率',
                                    data: data.cpu_data,
                                    borderColor: '#667eea',
                                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                    tension: 0.4
                                },
                                {
                                    label: '内存使用率',
                                    data: data.memory_data,
                                    borderColor: '#764ba2',
                                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                                    tension: 0.4
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100
                                }
                            }
                        }
                    });
                });
            
            // 服务图表
            fetch('/api/charts/services')
                .then(response => response.json())
                .then(data => {
                    if (serviceChart) {
                        serviceChart.destroy();
                    }
                    
                    const ctx = document.getElementById('service-chart').getContext('2d');
                    serviceChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.labels,
                            datasets: [{
                                label: '响应时间(ms)',
                                data: data.response_times,
                                backgroundColor: ['#667eea', '#764ba2', '#f093fb']
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                });
        }
        
        // 页面加载时初始化
        document.addEventListener('DOMContentLoaded', function() {
            refreshData();
            // 每30秒自动刷新
            setInterval(refreshData, 30000);
        });
    </script>
</body>
</html>
        """
    
    def _get_system_metrics(self) -> Dict:
        """获取系统指标"""
        if self.collector.system_metrics:
            latest = self.collector.system_metrics[-1]
            return {
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'disk_percent': latest.disk_percent,
                'network_sent': latest.network_sent,
                'network_recv': latest.network_recv,
                'active_processes': latest.active_processes
            }
        return {}
    
    def _get_service_metrics(self) -> List[Dict]:
        """获取服务指标"""
        return [
            {
                'service_name': name,
                'status': metrics.status,
                'response_time': metrics.response_time,
                'error_count': metrics.error_count,
                'request_count': metrics.request_count,
                'last_check': metrics.last_check.isoformat()
            }
            for name, metrics in self.collector.service_metrics.items()
        ]
    
    def _get_alerts(self, resolved: bool = False) -> List[Dict]:
        """获取告警"""
        alerts = [alert for alert in self.collector.alerts if alert.resolved == resolved]
        return [
            {
                'id': alert.id,
                'level': alert.level,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'service': alert.service,
                'resolved': alert.resolved
            }
            for alert in alerts
        ]
    
    def _generate_system_charts(self) -> Dict:
        """生成系统图表数据"""
        history = self.db_manager.get_system_metrics_history(1)  # 最近1小时
        
        if not history:
            return {'labels': [], 'cpu_data': [], 'memory_data': []}
        
        # 按时间排序
        history.sort(key=lambda x: x['timestamp'])
        
        labels = [item['timestamp'][11:16] for item in history]  # 只显示时间部分
        cpu_data = [item['cpu_percent'] for item in history]
        memory_data = [item['memory_percent'] for item in history]
        
        return {
            'labels': labels,
            'cpu_data': cpu_data,
            'memory_data': memory_data
        }
    
    def _generate_service_charts(self) -> Dict:
        """生成服务图表数据"""
        services = self.db_manager.get_service_metrics()
        
        labels = [service['service_name'] for service in services]
        response_times = [service['response_time'] for service in services]
        
        return {
            'labels': labels,
            'response_times': response_times
        }
    
    def start_monitoring(self):
        """开始监控"""
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        """监控循环"""
        services = [
            ('http://localhost:5000', 'web_interface'),
            ('http://localhost:8000', 'api_service'),
            ('http://localhost:9000', 'batch_processor')
        ]
        
        while not self.stop_monitoring:
            try:
                # 收集系统指标
                system_metrics = self.collector.collect_system_metrics()
                if system_metrics:
                    self.db_manager.save_system_metrics(system_metrics)
                
                # 收集服务指标
                for service_url, service_name in services:
                    service_metrics = self.collector.collect_service_metrics(service_url, service_name)
                    if service_metrics:
                        self.db_manager.save_service_metrics(service_metrics)
                
                # 检查告警
                new_alerts = self.collector.check_alerts()
                for alert in new_alerts:
                    self.db_manager.save_alert(alert)
                
                # 等待30秒
                time.sleep(30)
                
            except Exception as e:
                print(f"监控循环出错: {str(e)}")
                time.sleep(60)  # 出错时等待更长时间
    
    def run(self, host: str = '0.0.0.0', port: int = 7000, debug: bool = False):
        """运行监控仪表板"""
        print(f"启动监控仪表板: http://{host}:{port}")
        
        # 开始监控
        self.start_monitoring()
        
        # 启动Flask应用
        self.app.run(host=host, port=port, debug=debug)

def main():
    """主函数"""
    print("数学格式修复监控仪表板")
    print("=" * 50)
    
    # 创建监控仪表板
    dashboard = MonitoringDashboard()
    
    # 运行仪表板
    dashboard.run(host='localhost', port=7000)

if __name__ == '__main__':
    main() 