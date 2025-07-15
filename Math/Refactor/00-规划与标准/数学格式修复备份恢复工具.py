#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学格式修复备份恢复工具
提供完整的备份、恢复和版本管理功能
"""

import os
import sys
import json
import shutil
import hashlib
import zipfile
import tarfile
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import yaml
import requests

@dataclass
class BackupInfo:
    """备份信息"""
    id: str
    name: str
    description: str
    timestamp: datetime
    size: int
    file_count: int
    checksum: str
    version: str
    backup_type: str  # 'full', 'incremental', 'differential'
    status: str  # 'success', 'failed', 'in_progress'

@dataclass
class RestoreInfo:
    """恢复信息"""
    backup_id: str
    restore_timestamp: datetime
    target_path: str
    status: str  # 'success', 'failed', 'in_progress'
    restored_files: int
    errors: List[str]

class BackupManager:
    """备份管理器"""
    
    def __init__(self, backup_dir: str = 'backups', db_file: str = 'backup.db'):
        self.backup_dir = backup_dir
        self.db_file = db_file
        self.logger = self._setup_logger()
        self._init_backup_dir()
        self._init_database()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('BackupManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_backup_dir(self):
        """初始化备份目录"""
        os.makedirs(self.backup_dir, exist_ok=True)
        self.logger.info(f"备份目录: {self.backup_dir}")
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # 创建备份信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                timestamp TEXT NOT NULL,
                size INTEGER,
                file_count INTEGER,
                checksum TEXT,
                version TEXT,
                backup_type TEXT,
                status TEXT,
                file_path TEXT
            )
        ''')
        
        # 创建恢复信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS restores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_id TEXT,
                restore_timestamp TEXT,
                target_path TEXT,
                status TEXT,
                restored_files INTEGER,
                errors TEXT,
                FOREIGN KEY (backup_id) REFERENCES backups (id)
            )
        ''')
        
        # 创建文件索引表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backup_files (
                backup_id TEXT,
                file_path TEXT,
                file_size INTEGER,
                file_hash TEXT,
                PRIMARY KEY (backup_id, file_path),
                FOREIGN KEY (backup_id) REFERENCES backups (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_file_list(self, source_path: str, exclude_patterns: List[str] = None) -> List[str]:
        """获取文件列表"""
        if exclude_patterns is None:
            exclude_patterns = ['.git', '__pycache__', '.pyc', '.DS_Store', 'backups']
        
        file_list = []
        
        for root, dirs, files in os.walk(source_path):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in exclude_patterns]
            
            for file in files:
                file_path = os.path.join(root, file)
                # 排除不需要的文件
                if not any(pattern in file_path for pattern in exclude_patterns):
                    file_list.append(file_path)
        
        return file_list
    
    def _get_total_size(self, file_list: List[str]) -> int:
        """计算总大小"""
        total_size = 0
        for file_path in file_list:
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                continue
        return total_size
    
    def create_backup(self, source_path: str, backup_name: str, description: str = "", 
                     backup_type: str = "full", exclude_patterns: List[str] = None) -> BackupInfo:
        """创建备份"""
        backup_id = f"backup_{int(datetime.now().timestamp())}"
        backup_file = os.path.join(self.backup_dir, f"{backup_id}.zip")
        
        self.logger.info(f"开始创建备份: {backup_name}")
        
        try:
            # 获取文件列表
            file_list = self._get_file_list(source_path, exclude_patterns)
            total_size = self._get_total_size(file_list)
            
            # 创建备份信息
            backup_info = BackupInfo(
                id=backup_id,
                name=backup_name,
                description=description,
                timestamp=datetime.now(),
                size=0,
                file_count=len(file_list),
                checksum="",
                version="1.0.0",
                backup_type=backup_type,
                status="in_progress"
            )
            
            # 保存备份信息到数据库
            self._save_backup_info(backup_info, backup_file)
            
            # 创建ZIP文件
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_list:
                    try:
                        # 计算相对路径
                        rel_path = os.path.relpath(file_path, source_path)
                        zipf.write(file_path, rel_path)
                        
                        # 记录文件信息
                        self._save_file_info(backup_id, rel_path, file_path)
                        
                    except Exception as e:
                        self.logger.warning(f"备份文件失败 {file_path}: {str(e)}")
            
            # 计算最终大小和校验和
            final_size = os.path.getsize(backup_file)
            checksum = self._calculate_checksum(backup_file)
            
            # 更新备份信息
            backup_info.size = final_size
            backup_info.checksum = checksum
            backup_info.status = "success"
            
            # 更新数据库
            self._update_backup_info(backup_info)
            
            self.logger.info(f"备份创建成功: {backup_name} ({final_size} bytes)")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"备份创建失败: {str(e)}")
            backup_info.status = "failed"
            self._update_backup_info(backup_info)
            raise
    
    def _save_backup_info(self, backup_info: BackupInfo, file_path: str):
        """保存备份信息到数据库"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backups 
            (id, name, description, timestamp, size, file_count, checksum, version, backup_type, status, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            backup_info.id,
            backup_info.name,
            backup_info.description,
            backup_info.timestamp.isoformat(),
            backup_info.size,
            backup_info.file_count,
            backup_info.checksum,
            backup_info.version,
            backup_info.backup_type,
            backup_info.status,
            file_path
        ))
        
        conn.commit()
        conn.close()
    
    def _save_file_info(self, backup_id: str, rel_path: str, file_path: str):
        """保存文件信息"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        try:
            file_size = os.path.getsize(file_path)
            file_hash = self._calculate_checksum(file_path)
            
            cursor.execute('''
                INSERT INTO backup_files (backup_id, file_path, file_size, file_hash)
                VALUES (?, ?, ?, ?)
            ''', (backup_id, rel_path, file_size, file_hash))
            
            conn.commit()
        except Exception as e:
            self.logger.warning(f"保存文件信息失败 {file_path}: {str(e)}")
        finally:
            conn.close()
    
    def _update_backup_info(self, backup_info: BackupInfo):
        """更新备份信息"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE backups 
            SET size = ?, checksum = ?, status = ?
            WHERE id = ?
        ''', (backup_info.size, backup_info.checksum, backup_info.status, backup_info.id))
        
        conn.commit()
        conn.close()
    
    def list_backups(self) -> List[BackupInfo]:
        """列出所有备份"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM backups ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        backups = []
        for row in rows:
            backup = BackupInfo(
                id=row[0],
                name=row[1],
                description=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                size=row[4],
                file_count=row[5],
                checksum=row[6],
                version=row[7],
                backup_type=row[8],
                status=row[9]
            )
            backups.append(backup)
        
        return backups
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """获取备份信息"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM backups WHERE id = ?', (backup_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return BackupInfo(
                id=row[0],
                name=row[1],
                description=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                size=row[4],
                file_count=row[5],
                checksum=row[6],
                version=row[7],
                backup_type=row[8],
                status=row[9]
            )
        
        return None
    
    def restore_backup(self, backup_id: str, target_path: str, overwrite: bool = False) -> RestoreInfo:
        """恢复备份"""
        self.logger.info(f"开始恢复备份: {backup_id}")
        
        # 获取备份信息
        backup_info = self.get_backup_info(backup_id)
        if not backup_info:
            raise ValueError(f"备份不存在: {backup_id}")
        
        # 获取备份文件路径
        backup_file = os.path.join(self.backup_dir, f"{backup_id}.zip")
        if not os.path.exists(backup_file):
            raise FileNotFoundError(f"备份文件不存在: {backup_file}")
        
        # 创建恢复信息
        restore_info = RestoreInfo(
            backup_id=backup_id,
            restore_timestamp=datetime.now(),
            target_path=target_path,
            status="in_progress",
            restored_files=0,
            errors=[]
        )
        
        try:
            # 创建目标目录
            os.makedirs(target_path, exist_ok=True)
            
            # 检查目标目录是否为空
            if not overwrite and os.listdir(target_path):
                raise ValueError(f"目标目录不为空: {target_path}")
            
            # 解压备份文件
            restored_files = 0
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                # 验证校验和
                if backup_info.checksum != self._calculate_checksum(backup_file):
                    raise ValueError("备份文件校验和不匹配")
                
                # 解压文件
                for file_info in zipf.infolist():
                    try:
                        zipf.extract(file_info, target_path)
                        restored_files += 1
                    except Exception as e:
                        error_msg = f"解压文件失败 {file_info.filename}: {str(e)}"
                        restore_info.errors.append(error_msg)
                        self.logger.warning(error_msg)
            
            # 更新恢复信息
            restore_info.restored_files = restored_files
            restore_info.status = "success"
            
            # 保存恢复信息
            self._save_restore_info(restore_info)
            
            self.logger.info(f"备份恢复成功: {backup_id} -> {target_path}")
            return restore_info
            
        except Exception as e:
            error_msg = f"备份恢复失败: {str(e)}"
            restore_info.errors.append(error_msg)
            restore_info.status = "failed"
            self._save_restore_info(restore_info)
            self.logger.error(error_msg)
            raise
    
    def _save_restore_info(self, restore_info: RestoreInfo):
        """保存恢复信息"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO restores 
            (backup_id, restore_timestamp, target_path, status, restored_files, errors)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            restore_info.backup_id,
            restore_info.restore_timestamp.isoformat(),
            restore_info.target_path,
            restore_info.status,
            restore_info.restored_files,
            json.dumps(restore_info.errors)
        ))
        
        conn.commit()
        conn.close()
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        try:
            # 获取备份信息
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                raise ValueError(f"备份不存在: {backup_id}")
            
            # 删除备份文件
            backup_file = os.path.join(self.backup_dir, f"{backup_id}.zip")
            if os.path.exists(backup_file):
                os.remove(backup_file)
            
            # 删除数据库记录
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # 删除文件记录
            cursor.execute('DELETE FROM backup_files WHERE backup_id = ?', (backup_id,))
            
            # 删除恢复记录
            cursor.execute('DELETE FROM restores WHERE backup_id = ?', (backup_id,))
            
            # 删除备份记录
            cursor.execute('DELETE FROM backups WHERE id = ?', (backup_id,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"备份删除成功: {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"备份删除失败: {str(e)}")
            return False
    
    def verify_backup(self, backup_id: str) -> Tuple[bool, List[str]]:
        """验证备份完整性"""
        errors = []
        
        try:
            # 获取备份信息
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                errors.append(f"备份不存在: {backup_id}")
                return False, errors
            
            # 检查备份文件
            backup_file = os.path.join(self.backup_dir, f"{backup_id}.zip")
            if not os.path.exists(backup_file):
                errors.append(f"备份文件不存在: {backup_file}")
                return False, errors
            
            # 验证校验和
            current_checksum = self._calculate_checksum(backup_file)
            if current_checksum != backup_info.checksum:
                errors.append("备份文件校验和不匹配")
                return False, errors
            
            # 验证ZIP文件完整性
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    if zipf.testzip() is not None:
                        errors.append("ZIP文件损坏")
                        return False, errors
            except Exception as e:
                errors.append(f"ZIP文件验证失败: {str(e)}")
                return False, errors
            
            # 验证文件数量
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                file_count = len(zipf.namelist())
                if file_count != backup_info.file_count:
                    errors.append(f"文件数量不匹配: 期望{backup_info.file_count}, 实际{file_count}")
                    return False, errors
            
            return True, errors
            
        except Exception as e:
            errors.append(f"验证过程出错: {str(e)}")
            return False, errors
    
    def create_incremental_backup(self, source_path: str, base_backup_id: str, 
                                backup_name: str, description: str = "") -> BackupInfo:
        """创建增量备份"""
        # 获取基础备份信息
        base_backup = self.get_backup_info(base_backup_id)
        if not base_backup:
            raise ValueError(f"基础备份不存在: {base_backup_id}")
        
        # 获取基础备份的文件列表
        base_files = self._get_backup_files(base_backup_id)
        base_file_hashes = {file['path']: file['hash'] for file in base_files}
        
        # 获取当前文件列表
        current_files = self._get_file_list(source_path)
        
        # 找出变化的文件
        changed_files = []
        for file_path in current_files:
            rel_path = os.path.relpath(file_path, source_path)
            current_hash = self._calculate_checksum(file_path)
            
            if rel_path not in base_file_hashes or base_file_hashes[rel_path] != current_hash:
                changed_files.append(file_path)
        
        # 创建增量备份
        backup_id = f"incremental_{int(datetime.now().timestamp())}"
        backup_file = os.path.join(self.backup_dir, f"{backup_id}.zip")
        
        self.logger.info(f"创建增量备份: {backup_name} ({len(changed_files)} 个变化文件)")
        
        try:
            # 创建备份信息
            backup_info = BackupInfo(
                id=backup_id,
                name=backup_name,
                description=description,
                timestamp=datetime.now(),
                size=0,
                file_count=len(changed_files),
                checksum="",
                version="1.0.0",
                backup_type="incremental",
                status="in_progress"
            )
            
            # 保存备份信息
            self._save_backup_info(backup_info, backup_file)
            
            # 创建ZIP文件
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in changed_files:
                    try:
                        rel_path = os.path.relpath(file_path, source_path)
                        zipf.write(file_path, rel_path)
                        self._save_file_info(backup_id, rel_path, file_path)
                    except Exception as e:
                        self.logger.warning(f"备份文件失败 {file_path}: {str(e)}")
            
            # 更新备份信息
            final_size = os.path.getsize(backup_file)
            checksum = self._calculate_checksum(backup_file)
            
            backup_info.size = final_size
            backup_info.checksum = checksum
            backup_info.status = "success"
            
            self._update_backup_info(backup_info)
            
            self.logger.info(f"增量备份创建成功: {backup_name}")
            return backup_info
            
        except Exception as e:
            self.logger.error(f"增量备份创建失败: {str(e)}")
            backup_info.status = "failed"
            self._update_backup_info(backup_info)
            raise
    
    def _get_backup_files(self, backup_id: str) -> List[Dict]:
        """获取备份文件列表"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT file_path, file_size, file_hash FROM backup_files WHERE backup_id = ?', (backup_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'path': row[0],
                'size': row[1],
                'hash': row[2]
            }
            for row in rows
        ]
    
    def generate_backup_report(self, backup_id: str = None) -> str:
        """生成备份报告"""
        if backup_id:
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                return f"备份不存在: {backup_id}"
            
            report = f"""
备份详细信息
============
ID: {backup_info.id}
名称: {backup_info.name}
描述: {backup_info.description}
时间: {backup_info.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
大小: {backup_info.size:,} bytes
文件数: {backup_info.file_count}
类型: {backup_info.backup_type}
状态: {backup_info.status}
版本: {backup_info.version}
校验和: {backup_info.checksum}
"""
            
            # 验证备份
            is_valid, errors = self.verify_backup(backup_id)
            report += f"\n验证结果: {'通过' if is_valid else '失败'}"
            if errors:
                report += f"\n错误信息:\n"
                for error in errors:
                    report += f"  - {error}\n"
            
            return report
        
        else:
            # 生成所有备份的报告
            backups = self.list_backups()
            
            report = f"""
备份总览报告
============
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总备份数: {len(backups)}

备份列表:
"""
            
            total_size = 0
            for backup in backups:
                report += f"""
ID: {backup.id}
名称: {backup.name}
时间: {backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
大小: {backup.size:,} bytes
类型: {backup.backup_type}
状态: {backup.status}
"""
                total_size += backup.size
            
            report += f"\n总大小: {total_size:,} bytes"
            
            return report

def main():
    """主函数 - 演示备份恢复工具"""
    print("数学格式修复备份恢复工具")
    print("=" * 50)
    
    # 创建备份管理器
    backup_manager = BackupManager()
    
    # 示例：创建备份
    try:
        print("创建项目备份...")
        backup_info = backup_manager.create_backup(
            source_path=".",
            backup_name="数学格式修复项目完整备份",
            description="包含所有项目文件和配置的完整备份",
            backup_type="full"
        )
        print(f"备份创建成功: {backup_info.name}")
        
        # 列出所有备份
        print("\n所有备份:")
        backups = backup_manager.list_backups()
        for backup in backups:
            print(f"  - {backup.name} ({backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # 生成报告
        print("\n备份报告:")
        report = backup_manager.generate_backup_report()
        print(report)
        
        # 验证备份
        print("\n验证备份...")
        is_valid, errors = backup_manager.verify_backup(backup_info.id)
        print(f"验证结果: {'通过' if is_valid else '失败'}")
        if errors:
            for error in errors:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"备份操作失败: {str(e)}")

if __name__ == '__main__':
    main() 