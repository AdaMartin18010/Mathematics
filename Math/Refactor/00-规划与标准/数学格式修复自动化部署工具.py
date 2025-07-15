#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复自动化部署工具
提供一键部署、环境配置和部署验证功能
"""

import os
import sys
import json
import subprocess
import logging
import shutil
import platform
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml
import requests

@dataclass
class DeploymentConfig:
    """部署配置"""
    project_name: str
    version: str
    environment: str  # 'development', 'staging', 'production'
    python_version: str
    dependencies: List[str]
    services: List[str]
    ports: Dict[str, int]
    volumes: Dict[str, str]
    environment_vars: Dict[str, str]

@dataclass
class DeploymentResult:
    """部署结果"""
    success: bool
    timestamp: datetime
    environment: str
    services_deployed: List[str]
    errors: List[str]
    warnings: List[str]
    deployment_time: float
    status_checks: Dict[str, bool]

class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, config_file: str = 'deployment_config.yaml'):
        self.config_file = config_file
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('DeploymentManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> DeploymentConfig:
        """加载部署配置"""
        if not os.path.exists(self.config_file):
            # 创建默认配置
            default_config = {
                'project_name': 'math-format-fix',
                'version': '1.0.0',
                'environment': 'development',
                'python_version': '3.8+',
                'dependencies': [
                    'flask',
                    'requests',
                    'psutil',
                    'matplotlib',
                    'numpy'
                ],
                'services': [
                    'web_interface',
                    'api_service',
                    'batch_processor'
                ],
                'ports': {
                    'web_interface': 5000,
                    'api_service': 8000,
                    'batch_processor': 9000
                },
                'volumes': {
                    'data': './data',
                    'logs': './logs',
                    'config': './config'
                },
                'environment_vars': {
                    'FLASK_ENV': 'development',
                    'DEBUG': 'True',
                    'LOG_LEVEL': 'INFO'
                }
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            self.logger.info(f"创建默认配置文件: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return DeploymentConfig(**config_data)
    
    def check_environment(self) -> Tuple[bool, List[str]]:
        """检查部署环境"""
        issues = []
        
        # 检查Python版本
        python_version = sys.version_info
        required_version = tuple(map(int, self.config.python_version.replace('+', '').split('.')))
        
        if python_version < required_version:
            issues.append(f"Python版本过低: 需要{self.config.python_version}，当前{sys.version}")
        
        # 检查必要工具
        required_tools = ['pip', 'git']
        for tool in required_tools:
            if not shutil.which(tool):
                issues.append(f"缺少必要工具: {tool}")
        
        # 检查磁盘空间
        try:
            disk_usage = shutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 1.0:  # 需要至少1GB空间
                issues.append(f"磁盘空间不足: 可用{free_gb:.1f}GB，需要至少1GB")
        except Exception as e:
            issues.append(f"无法检查磁盘空间: {str(e)}")
        
        # 检查网络连接
        try:
            response = requests.get('https://pypi.org/simple/', timeout=5)
            if response.status_code != 200:
                issues.append("无法连接到PyPI，可能影响依赖安装")
        except Exception:
            issues.append("网络连接异常，可能影响依赖安装")
        
        return len(issues) == 0, issues
    
    def install_dependencies(self) -> Tuple[bool, List[str]]:
        """安装项目依赖"""
        errors = []
        
        try:
            # 升级pip
            self.logger.info("升级pip...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # 安装依赖
            self.logger.info("安装项目依赖...")
            for dependency in self.config.dependencies:
                try:
                    self.logger.info(f"安装 {dependency}...")
                    subprocess.run([sys.executable, '-m', 'pip', 'install', dependency], 
                                 check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    error_msg = f"安装 {dependency} 失败: {e.stderr.decode()}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # 安装项目本身
            self.logger.info("安装项目...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], 
                         check=True, capture_output=True)
            
        except Exception as e:
            errors.append(f"依赖安装过程出错: {str(e)}")
            self.logger.error(f"依赖安装失败: {str(e)}")
        
        return len(errors) == 0, errors
    
    def create_directories(self) -> Tuple[bool, List[str]]:
        """创建必要的目录"""
        errors = []
        
        try:
            directories = [
                'data',
                'logs', 
                'config',
                'temp',
                'backup'
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"创建目录: {directory}")
            
            # 创建日志文件
            log_file = 'logs/deployment.log'
            with open(log_file, 'w') as f:
                f.write(f"部署日志 - {datetime.now()}\n")
            
        except Exception as e:
            error_msg = f"创建目录失败: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        return len(errors) == 0, errors
    
    def setup_environment_variables(self) -> Tuple[bool, List[str]]:
        """设置环境变量"""
        errors = []
        
        try:
            # 设置环境变量
            for key, value in self.config.environment_vars.items():
                os.environ[key] = value
                self.logger.info(f"设置环境变量: {key}={value}")
            
            # 创建.env文件
            env_file = '.env'
            with open(env_file, 'w', encoding='utf-8') as f:
                for key, value in self.config.environment_vars.items():
                    f.write(f"{key}={value}\n")
            
            self.logger.info(f"创建环境变量文件: {env_file}")
            
        except Exception as e:
            error_msg = f"设置环境变量失败: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        return len(errors) == 0, errors
    
    def deploy_services(self) -> Tuple[bool, List[str], List[str]]:
        """部署服务"""
        errors = []
        warnings = []
        deployed_services = []
        
        for service in self.config.services:
            try:
                self.logger.info(f"部署服务: {service}")
                
                if service == 'web_interface':
                    success = self._deploy_web_interface()
                elif service == 'api_service':
                    success = self._deploy_api_service()
                elif service == 'batch_processor':
                    success = self._deploy_batch_processor()
                else:
                    warnings.append(f"未知服务: {service}")
                    continue
                
                if success:
                    deployed_services.append(service)
                    self.logger.info(f"服务 {service} 部署成功")
                else:
                    errors.append(f"服务 {service} 部署失败")
                
            except Exception as e:
                error_msg = f"部署服务 {service} 时出错: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        return len(errors) == 0, errors, warnings
    
    def _deploy_web_interface(self) -> bool:
        """部署Web界面"""
        try:
            # 检查Web界面文件
            web_file = '数学格式修复Web界面.py'
            if not os.path.exists(web_file):
                self.logger.error(f"Web界面文件不存在: {web_file}")
                return False
            
            # 启动Web服务
            port = self.config.ports['web_interface']
            cmd = [sys.executable, web_file]
            
            # 在后台启动服务
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 等待服务启动
            import time
            time.sleep(3)
            
            # 检查服务是否启动成功
            try:
                response = requests.get(f'http://localhost:{port}/health', timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"Web界面部署成功，端口: {port}")
                    return True
                else:
                    self.logger.error(f"Web界面健康检查失败: {response.status_code}")
                    return False
            except requests.RequestException:
                self.logger.error("Web界面无法访问")
                return False
                
        except Exception as e:
            self.logger.error(f"部署Web界面失败: {str(e)}")
            return False
    
    def _deploy_api_service(self) -> bool:
        """部署API服务"""
        try:
            # 检查API服务文件
            api_file = '数学格式修复API服务.py'
            if not os.path.exists(api_file):
                self.logger.error(f"API服务文件不存在: {api_file}")
                return False
            
            # 启动API服务
            port = self.config.ports['api_service']
            cmd = [sys.executable, api_file]
            
            # 在后台启动服务
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 等待服务启动
            import time
            time.sleep(3)
            
            # 检查服务是否启动成功
            try:
                response = requests.get(f'http://localhost:{port}/health', timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"API服务部署成功，端口: {port}")
                    return True
                else:
                    self.logger.error(f"API服务健康检查失败: {response.status_code}")
                    return False
            except requests.RequestException:
                self.logger.error("API服务无法访问")
                return False
                
        except Exception as e:
            self.logger.error(f"部署API服务失败: {str(e)}")
            return False
    
    def _deploy_batch_processor(self) -> bool:
        """部署批量处理器"""
        try:
            # 检查批量处理文件
            batch_file = '数学格式修复批量处理工具.py'
            if not os.path.exists(batch_file):
                self.logger.error(f"批量处理文件不存在: {batch_file}")
                return False
            
            # 批量处理器通常不需要持续运行，这里只检查文件存在性
            self.logger.info("批量处理器文件检查通过")
            return True
                
        except Exception as e:
            self.logger.error(f"部署批量处理器失败: {str(e)}")
            return False
    
    def verify_deployment(self) -> Dict[str, bool]:
        """验证部署"""
        status_checks = {}
        
        # 检查服务状态
        for service, port in self.config.ports.items():
            try:
                response = requests.get(f'http://localhost:{port}/health', timeout=5)
                status_checks[f"{service}_health"] = response.status_code == 200
            except requests.RequestException:
                status_checks[f"{service}_health"] = False
        
        # 检查文件系统
        for volume_name, volume_path in self.config.volumes.items():
            status_checks[f"{volume_name}_accessible"] = os.path.exists(volume_path)
        
        # 检查环境变量
        for env_var in self.config.environment_vars:
            status_checks[f"{env_var}_set"] = env_var in os.environ
        
        return status_checks
    
    def deploy(self) -> DeploymentResult:
        """执行完整部署"""
        start_time = datetime.now()
        errors = []
        warnings = []
        deployed_services = []
        
        self.logger.info("开始部署数学格式修复项目...")
        
        # 1. 检查环境
        self.logger.info("检查部署环境...")
        env_ok, env_issues = self.check_environment()
        if not env_ok:
            errors.extend(env_issues)
            self.logger.error("环境检查失败")
        
        # 2. 安装依赖
        if env_ok:
            self.logger.info("安装项目依赖...")
            deps_ok, deps_errors = self.install_dependencies()
            if not deps_ok:
                errors.extend(deps_errors)
                self.logger.error("依赖安装失败")
        
        # 3. 创建目录
        if env_ok and deps_ok:
            self.logger.info("创建必要目录...")
            dirs_ok, dirs_errors = self.create_directories()
            if not dirs_ok:
                errors.extend(dirs_errors)
                self.logger.error("目录创建失败")
        
        # 4. 设置环境变量
        if env_ok and deps_ok and dirs_ok:
            self.logger.info("设置环境变量...")
            env_vars_ok, env_vars_errors = self.setup_environment_variables()
            if not env_vars_ok:
                errors.extend(env_vars_errors)
                self.logger.error("环境变量设置失败")
        
        # 5. 部署服务
        if env_ok and deps_ok and dirs_ok and env_vars_ok:
            self.logger.info("部署服务...")
            services_ok, services_errors, services_warnings = self.deploy_services()
            if not services_ok:
                errors.extend(services_errors)
                self.logger.error("服务部署失败")
            warnings.extend(services_warnings)
        
        # 6. 验证部署
        self.logger.info("验证部署...")
        status_checks = self.verify_deployment()
        
        # 计算部署时间
        deployment_time = (datetime.now() - start_time).total_seconds()
        
        # 确定部署成功状态
        success = len(errors) == 0 and len(deployed_services) > 0
        
        result = DeploymentResult(
            success=success,
            timestamp=datetime.now(),
            environment=self.config.environment,
            services_deployed=deployed_services,
            errors=errors,
            warnings=warnings,
            deployment_time=deployment_time,
            status_checks=status_checks
        )
        
        # 记录部署结果
        self._log_deployment_result(result)
        
        return result
    
    def _log_deployment_result(self, result: DeploymentResult):
        """记录部署结果"""
        log_file = 'logs/deployment.log'
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== 部署结果 ===\n")
            f.write(f"时间: {result.timestamp}\n")
            f.write(f"环境: {result.environment}\n")
            f.write(f"成功: {result.success}\n")
            f.write(f"部署时间: {result.deployment_time:.2f}秒\n")
            f.write(f"部署服务: {', '.join(result.services_deployed)}\n")
            
            if result.errors:
                f.write(f"错误:\n")
                for error in result.errors:
                    f.write(f"  - {error}\n")
            
            if result.warnings:
                f.write(f"警告:\n")
                for warning in result.warnings:
                    f.write(f"  - {warning}\n")
            
            f.write(f"状态检查:\n")
            for check, status in result.status_checks.items():
                f.write(f"  - {check}: {'通过' if status else '失败'}\n")
    
    def generate_deployment_report(self, result: DeploymentResult, output_file: str = None) -> str:
        """生成部署报告"""
        report = f"""
数学格式修复项目部署报告
生成时间: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

部署概览:
  环境: {result.environment}
  状态: {'成功' if result.success else '失败'}
  部署时间: {result.deployment_time:.2f}秒
  部署服务: {', '.join(result.services_deployed)}

状态检查:
"""
        
        for check, status in result.status_checks.items():
            report += f"  {check}: {'✓' if status else '✗'}\n"
        
        if result.errors:
            report += f"\n错误信息:\n"
            for error in result.errors:
                report += f"  - {error}\n"
        
        if result.warnings:
            report += f"\n警告信息:\n"
            for warning in result.warnings:
                report += f"  - {warning}\n"
        
        report += f"""
部署建议:
"""
        
        if result.success:
            report += "  - 部署成功，可以开始使用服务\n"
            report += "  - 建议定期检查服务状态\n"
            report += "  - 建议配置监控和日志管理\n"
        else:
            report += "  - 检查错误信息并修复问题\n"
            report += "  - 重新运行部署脚本\n"
            report += "  - 联系技术支持\n"
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"部署报告已保存: {output_file}")
        
        return report

class DockerDeployer:
    """Docker部署器"""
    
    def __init__(self):
        self.logger = logging.getLogger('DockerDeployer')
    
    def create_dockerfile(self, output_file: str = 'Dockerfile'):
        """创建Dockerfile"""
        dockerfile_content = """
FROM python:3.8-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要目录
RUN mkdir -p data logs config temp backup

# 暴露端口
EXPOSE 5000 8000 9000

# 设置环境变量
ENV FLASK_ENV=production
ENV DEBUG=False
ENV LOG_LEVEL=INFO

# 启动命令
CMD ["python", "数学格式修复Web界面.py"]
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"Dockerfile已创建: {output_file}")
    
    def create_docker_compose(self, output_file: str = 'docker-compose.yml'):
        """创建docker-compose.yml"""
        compose_content = """
version: '3.8'

services:
  web-interface:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - FLASK_ENV=production
      - DEBUG=False
    restart: unless-stopped

  api-service:
    build: .
    command: ["python", "数学格式修复API服务.py"]
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - DEBUG=False
    restart: unless-stopped

  batch-processor:
    build: .
    command: ["python", "数学格式修复批量处理工具.py"]
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(compose_content)
        
        self.logger.info(f"docker-compose.yml已创建: {output_file}")
    
    def deploy_with_docker(self) -> bool:
        """使用Docker部署"""
        try:
            # 创建Dockerfile
            self.create_dockerfile()
            
            # 创建docker-compose.yml
            self.create_docker_compose()
            
            # 构建镜像
            self.logger.info("构建Docker镜像...")
            subprocess.run(['docker-compose', 'build'], check=True)
            
            # 启动服务
            self.logger.info("启动Docker服务...")
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            
            self.logger.info("Docker部署成功")
            return True
            
        except Exception as e:
            self.logger.error(f"Docker部署失败: {str(e)}")
            return False

def main():
    """主函数 - 演示自动化部署工具"""
    print("数学格式修复项目自动化部署工具")
    print("=" * 50)
    
    # 创建部署管理器
    deployer = DeploymentManager()
    
    # 执行部署
    print("开始部署...")
    result = deployer.deploy()
    
    # 生成报告
    report = deployer.generate_deployment_report(result, "deployment_report.txt")
    print(report)
    
    # 如果部署成功，提供Docker选项
    if result.success:
        print("\n是否使用Docker部署? (y/n): ", end="")
        choice = input().lower()
        
        if choice == 'y':
            docker_deployer = DockerDeployer()
            if docker_deployer.deploy_with_docker():
                print("Docker部署成功!")
            else:
                print("Docker部署失败!")
    
    print("\n部署完成!")

if __name__ == '__main__':
    main() 