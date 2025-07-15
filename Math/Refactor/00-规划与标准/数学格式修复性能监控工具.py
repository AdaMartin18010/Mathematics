#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复性能监控工具
提供实时性能监控、资源使用统计和性能优化建议
"""

import time
import psutil
import threading
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_threads: int
    processing_files: int
    queue_size: int
    response_time_ms: float
    error_count: int
    success_count: int

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.logger = self._setup_logger()
        
        # 性能阈值
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_ms': 1000.0,
            'error_rate': 5.0
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('PerformanceMonitor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_monitoring(self):
        """开始性能监控"""
        if self.is_monitoring:
            self.logger.warning("性能监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 检查性能警告
                self._check_performance_warnings(metrics)
                
                # 限制历史记录大小
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        # 系统资源使用
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # 进程信息
        current_process = psutil.Process()
        memory_used_mb = current_process.memory_info().rss / 1024 / 1024
        
        # 线程信息
        active_threads = threading.active_count()
        
        # 应用特定指标（模拟）
        processing_files = self._get_processing_files_count()
        queue_size = self._get_queue_size()
        response_time = self._get_average_response_time()
        error_count = self._get_error_count()
        success_count = self._get_success_count()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            disk_io_read_mb=disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
            disk_io_write_mb=disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
            network_sent_mb=network_io.bytes_sent / 1024 / 1024 if network_io else 0,
            network_recv_mb=network_io.bytes_recv / 1024 / 1024 if network_io else 0,
            active_threads=active_threads,
            processing_files=processing_files,
            queue_size=queue_size,
            response_time_ms=response_time,
            error_count=error_count,
            success_count=success_count
        )
    
    def _get_processing_files_count(self) -> int:
        """获取正在处理的文件数量（模拟）"""
        return len([t for t in threading.enumerate() if 'process' in t.name.lower()])
    
    def _get_queue_size(self) -> int:
        """获取队列大小（模拟）"""
        return 0  # 实际应用中需要从队列管理器获取
    
    def _get_average_response_time(self) -> float:
        """获取平均响应时间（模拟）"""
        return 150.0  # 模拟150ms平均响应时间
    
    def _get_error_count(self) -> int:
        """获取错误计数（模拟）"""
        return 0  # 实际应用中需要从错误统计器获取
    
    def _get_success_count(self) -> int:
        """获取成功计数（模拟）"""
        return 100  # 模拟100个成功处理
    
    def _check_performance_warnings(self, metrics: PerformanceMetrics):
        """检查性能警告"""
        warnings = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            warnings.append(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            warnings.append(f"内存使用率过高: {metrics.memory_percent:.1f}%")
        
        if metrics.response_time_ms > self.thresholds['response_time_ms']:
            warnings.append(f"响应时间过长: {metrics.response_time_ms:.1f}ms")
        
        if metrics.error_count > 0:
            total_requests = metrics.success_count + metrics.error_count
            error_rate = (metrics.error_count / total_requests) * 100
            if error_rate > self.thresholds['error_rate']:
                warnings.append(f"错误率过高: {error_rate:.1f}%")
        
        if warnings:
            self.logger.warning(f"性能警告: {'; '.join(warnings)}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前性能指标"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        return asdict(latest)
    
    def get_performance_summary(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # 计算统计信息
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        response_times = [m.response_time_ms for m in recent_metrics]
        
        total_requests = sum(m.success_count + m.error_count for m in recent_metrics)
        total_errors = sum(m.error_count for m in recent_metrics)
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'duration_minutes': duration_minutes,
            'metrics_count': len(recent_metrics),
            'cpu': {
                'average': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values)
            },
            'memory': {
                'average': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values)
            },
            'response_time': {
                'average': np.mean(response_times),
                'max': np.max(response_times),
                'min': np.min(response_times)
            },
            'requests': {
                'total': total_requests,
                'success': sum(m.success_count for m in recent_metrics),
                'errors': total_errors,
                'error_rate': error_rate
            }
        }
    
    def generate_performance_report(self, output_file: str = None) -> str:
        """生成性能报告"""
        if not self.metrics_history:
            return "无性能数据可用"
        
        # 获取性能摘要
        summary = self.get_performance_summary()
        
        # 生成报告内容
        report = f"""
数学格式修复性能监控报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
监控时长: {summary.get('duration_minutes', 0)} 分钟
数据点数: {summary.get('metrics_count', 0)}

性能指标摘要:
CPU使用率:
  平均: {summary.get('cpu', {}).get('average', 0):.1f}%
  最高: {summary.get('cpu', {}).get('max', 0):.1f}%
  最低: {summary.get('cpu', {}).get('min', 0):.1f}%

内存使用率:
  平均: {summary.get('memory', {}).get('average', 0):.1f}%
  最高: {summary.get('memory', {}).get('max', 0):.1f}%
  最低: {summary.get('memory', {}).get('min', 0):.1f}%

响应时间:
  平均: {summary.get('response_time', {}).get('average', 0):.1f}ms
  最高: {summary.get('response_time', {}).get('max', 0):.1f}ms
  最低: {summary.get('response_time', {}).get('min', 0):.1f}ms

请求统计:
  总请求数: {summary.get('requests', {}).get('total', 0)}
  成功请求: {summary.get('requests', {}).get('success', 0)}
  错误请求: {summary.get('requests', {}).get('errors', 0)}
  错误率: {summary.get('requests', {}).get('error_rate', 0):.1f}%

性能建议:
"""
        
        # 添加性能建议
        recommendations = self._generate_recommendations(summary)
        report += recommendations
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"性能报告已保存: {output_file}")
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> str:
        """生成性能建议"""
        recommendations = []
        
        cpu_avg = summary.get('cpu', {}).get('average', 0)
        memory_avg = summary.get('memory', {}).get('average', 0)
        response_avg = summary.get('response_time', {}).get('average', 0)
        error_rate = summary.get('requests', {}).get('error_rate', 0)
        
        if cpu_avg > 70:
            recommendations.append("- CPU使用率较高，建议优化算法或增加处理节点")
        
        if memory_avg > 80:
            recommendations.append("- 内存使用率较高，建议优化内存使用或增加内存")
        
        if response_avg > 500:
            recommendations.append("- 响应时间较长，建议优化处理逻辑或使用缓存")
        
        if error_rate > 2:
            recommendations.append("- 错误率较高，建议检查错误处理逻辑")
        
        if not recommendations:
            recommendations.append("- 性能表现良好，无需特殊优化")
        
        return '\n'.join(recommendations)
    
    def plot_performance_charts(self, output_dir: str = "performance_charts"):
        """生成性能图表"""
        if not self.metrics_history:
            self.logger.warning("无性能数据可用于生成图表")
            return
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        timestamps = [m.timestamp for m in self.metrics_history]
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        response_times = [m.response_time_ms for m in self.metrics_history]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('数学格式修复性能监控图表', fontsize=16)
        
        # CPU使用率
        axes[0, 0].plot(timestamps, cpu_values, 'b-', linewidth=1)
        axes[0, 0].set_title('CPU使用率')
        axes[0, 0].set_ylabel('使用率 (%)')
        axes[0, 0].grid(True)
        
        # 内存使用率
        axes[0, 1].plot(timestamps, memory_values, 'r-', linewidth=1)
        axes[0, 1].set_title('内存使用率')
        axes[0, 1].set_ylabel('使用率 (%)')
        axes[0, 1].grid(True)
        
        # 响应时间
        axes[1, 0].plot(timestamps, response_times, 'g-', linewidth=1)
        axes[1, 0].set_title('响应时间')
        axes[1, 0].set_ylabel('时间 (ms)')
        axes[1, 0].grid(True)
        
        # 处理文件数量
        file_counts = [m.processing_files for m in self.metrics_history]
        axes[1, 1].plot(timestamps, file_counts, 'm-', linewidth=1)
        axes[1, 1].set_title('正在处理的文件数量')
        axes[1, 1].set_ylabel('文件数量')
        axes[1, 1].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        chart_file = os.path.join(output_dir, f"performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"性能图表已保存: {chart_file}")
    
    def export_metrics(self, output_file: str):
        """导出性能指标数据"""
        if not self.metrics_history:
            self.logger.warning("无性能数据可导出")
            return
        
        # 转换为可序列化的格式
        export_data = []
        for metrics in self.metrics_history:
            export_data.append({
                'timestamp': metrics.timestamp.isoformat(),
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'memory_used_mb': metrics.memory_used_mb,
                'disk_io_read_mb': metrics.disk_io_read_mb,
                'disk_io_write_mb': metrics.disk_io_write_mb,
                'network_sent_mb': metrics.network_sent_mb,
                'network_recv_mb': metrics.network_recv_mb,
                'active_threads': metrics.active_threads,
                'processing_files': metrics.processing_files,
                'queue_size': metrics.queue_size,
                'response_time_ms': metrics.response_time_ms,
                'error_count': metrics.error_count,
                'success_count': metrics.success_count
            })
        
        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"性能指标已导出: {output_file}")

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger('PerformanceOptimizer')
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能并提供优化建议"""
        summary = self.monitor.get_performance_summary()
        
        analysis = {
            'performance_score': self._calculate_performance_score(summary),
            'bottlenecks': self._identify_bottlenecks(summary),
            'optimization_suggestions': self._generate_optimization_suggestions(summary),
            'resource_usage': self._analyze_resource_usage(summary)
        }
        
        return analysis
    
    def _calculate_performance_score(self, summary: Dict[str, Any]) -> float:
        """计算性能评分 (0-100)"""
        score = 100.0
        
        # CPU评分
        cpu_avg = summary.get('cpu', {}).get('average', 0)
        if cpu_avg > 80:
            score -= 20
        elif cpu_avg > 60:
            score -= 10
        
        # 内存评分
        memory_avg = summary.get('memory', {}).get('average', 0)
        if memory_avg > 85:
            score -= 20
        elif memory_avg > 70:
            score -= 10
        
        # 响应时间评分
        response_avg = summary.get('response_time', {}).get('average', 0)
        if response_avg > 1000:
            score -= 20
        elif response_avg > 500:
            score -= 10
        
        # 错误率评分
        error_rate = summary.get('requests', {}).get('error_rate', 0)
        if error_rate > 5:
            score -= 15
        elif error_rate > 2:
            score -= 5
        
        return max(0, score)
    
    def _identify_bottlenecks(self, summary: Dict[str, Any]) -> List[str]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        cpu_avg = summary.get('cpu', {}).get('average', 0)
        memory_avg = summary.get('memory', {}).get('average', 0)
        response_avg = summary.get('response_time', {}).get('average', 0)
        error_rate = summary.get('requests', {}).get('error_rate', 0)
        
        if cpu_avg > 80:
            bottlenecks.append("CPU使用率过高")
        
        if memory_avg > 85:
            bottlenecks.append("内存使用率过高")
        
        if response_avg > 1000:
            bottlenecks.append("响应时间过长")
        
        if error_rate > 5:
            bottlenecks.append("错误率过高")
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, summary: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        cpu_avg = summary.get('cpu', {}).get('average', 0)
        memory_avg = summary.get('memory', {}).get('average', 0)
        response_avg = summary.get('response_time', {}).get('average', 0)
        error_rate = summary.get('requests', {}).get('error_rate', 0)
        
        if cpu_avg > 70:
            suggestions.extend([
                "优化算法复杂度",
                "增加处理节点",
                "使用并行处理",
                "实现任务队列"
            ])
        
        if memory_avg > 80:
            suggestions.extend([
                "优化内存使用",
                "增加系统内存",
                "实现内存缓存",
                "使用流式处理"
            ])
        
        if response_avg > 500:
            suggestions.extend([
                "优化处理逻辑",
                "使用缓存机制",
                "异步处理",
                "负载均衡"
            ])
        
        if error_rate > 2:
            suggestions.extend([
                "改进错误处理",
                "增加重试机制",
                "优化输入验证",
                "完善日志记录"
            ])
        
        return list(set(suggestions))  # 去重
    
    def _analyze_resource_usage(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """分析资源使用情况"""
        return {
            'cpu_usage': {
                'current': summary.get('cpu', {}).get('average', 0),
                'status': 'high' if summary.get('cpu', {}).get('average', 0) > 70 else 'normal'
            },
            'memory_usage': {
                'current': summary.get('memory', {}).get('average', 0),
                'status': 'high' if summary.get('memory', {}).get('average', 0) > 80 else 'normal'
            },
            'response_time': {
                'current': summary.get('response_time', {}).get('average', 0),
                'status': 'slow' if summary.get('response_time', {}).get('average', 0) > 500 else 'normal'
            },
            'error_rate': {
                'current': summary.get('requests', {}).get('error_rate', 0),
                'status': 'high' if summary.get('requests', {}).get('error_rate', 0) > 2 else 'normal'
            }
        }

def main():
    """主函数 - 演示性能监控工具"""
    # 创建性能监控器
    monitor = PerformanceMonitor(monitoring_interval=2.0)
    
    try:
        print("启动性能监控...")
        monitor.start_monitoring()
        
        # 模拟运行一段时间
        print("监控运行中... (按Ctrl+C停止)")
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n停止性能监控...")
        monitor.stop_monitoring()
        
        # 生成报告
        print("\n生成性能报告...")
        report = monitor.generate_performance_report("performance_report.txt")
        print(report)
        
        # 生成图表
        print("\n生成性能图表...")
        monitor.plot_performance_charts()
        
        # 导出数据
        print("\n导出性能数据...")
        monitor.export_metrics("performance_metrics.json")
        
        # 性能优化分析
        print("\n性能优化分析...")
        optimizer = PerformanceOptimizer(monitor)
        analysis = optimizer.analyze_performance()
        
        print(f"性能评分: {analysis['performance_score']:.1f}/100")
        print(f"性能瓶颈: {', '.join(analysis['bottlenecks']) if analysis['bottlenecks'] else '无'}")
        print(f"优化建议: {', '.join(analysis['optimization_suggestions'][:3])}")
        
        print("\n性能监控完成!")

if __name__ == '__main__':
    main() 