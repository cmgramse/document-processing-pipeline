"""
Resource monitoring module.

Monitors system resources and enforces resource constraints.
"""

import logging
import psutil
import time
from typing import Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import json
import shutil
import os
import atexit

from ..config.settings import settings
from .error_handler import ResourceError, error_store

logger = logging.getLogger(__name__)

@dataclass
class ResourceStats:
    """Resource statistics data class."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    disk_io_read_bytes: int = 0
    disk_io_write_bytes: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

class ResourceMonitor:
    """Monitors and manages system resources."""
    
    def __init__(self):
        """Initialize resource monitor."""
        data_dir = Path('./data')
        data_dir.mkdir(exist_ok=True)
        self.db_path = data_dir / "resources.db"
        self._init_db()
        self._last_disk_io = psutil.disk_io_counters()
        self._last_network_io = psutil.net_io_counters()
        self._last_check = datetime.now()
        self._temp_dirs = set()
        self._open_files = set()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def register_temp_dir(self, path: str) -> None:
        """Register a temporary directory for cleanup."""
        self._temp_dirs.add(Path(path))
    
    def register_open_file(self, path: str) -> None:
        """Register an open file for cleanup."""
        self._open_files.add(Path(path))
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Starting resource cleanup...")
        
        # Clean up temporary directories
        for temp_dir in self._temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up directory {temp_dir}: {e}")
        
        # Clean up open files
        for file_path in self._open_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")
        
        # Clean up old statistics
        self._cleanup_old_stats()
        
        # Clean up database connections
        try:
            self._cleanup_db_connections()
        except Exception as e:
            logger.error(f"Error cleaning up database connections: {e}")
        
        logger.info("Resource cleanup completed")
    
    def _cleanup_old_stats(self) -> None:
        """Clean up old statistics from database."""
        try:
            retention_days = settings.resources.stats_retention_days
            if not retention_days:
                return
            
            conn = sqlite3.connect(str(self.db_path))
            try:
                c = conn.cursor()
                c.execute(
                    """
                    DELETE FROM resource_stats
                    WHERE timestamp < datetime('now', ?)
                    """,
                    (f"-{retention_days} days",)
                )
                conn.commit()
                
                if c.rowcount > 0:
                    logger.info(f"Cleaned up {c.rowcount} old resource statistics")
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error cleaning up old statistics: {e}")
    
    def _cleanup_db_connections(self) -> None:
        """Clean up database connections."""
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                if proc.pid == os.getpid():
                    for file in proc.open_files():
                        if str(self.db_path) in file.path:
                            logger.warning(
                                f"Found open database connection: {file.path}"
                            )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Get network usage
            network = psutil.net_io_counters()
            
            # Get process information
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Initialize disk I/O values
            disk_io_read_mb = 0
            disk_io_write_mb = 0
            if disk_io:  # Check if disk_io is not None
                disk_io_read_mb = disk_io.read_bytes / (1024 * 1024)
                disk_io_write_mb = disk_io.write_bytes / (1024 * 1024)
            
            return {
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / (1024 * 1024),
                    'disk_percent': disk.percent,
                    'disk_free_mb': disk.free / (1024 * 1024),
                    'disk_io_read_mb': disk_io_read_mb,
                    'disk_io_write_mb': disk_io_write_mb,
                    'network_sent_mb': network.bytes_sent / (1024 * 1024),
                    'network_recv_mb': network.bytes_recv / (1024 * 1024)
                },
                'process': {
                    'memory_mb': process_memory.rss / (1024 * 1024),
                    'cpu_percent': process.cpu_percent(),
                    'open_files': len(process.open_files()),
                    'threads': process.num_threads()
                }
            }
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {}
    
    def check_resource_leaks(self) -> List[Dict[str, Any]]:
        """Check for potential resource leaks."""
        leaks = []
        process = psutil.Process()
        
        # Check for file descriptor leaks
        try:
            open_files = process.open_files()
            if len(open_files) > settings.resources.max_open_files:
                leaks.append({
                    'type': 'file_descriptors',
                    'current': len(open_files),
                    'limit': settings.resources.max_open_files,
                    'details': [str(f.path) for f in open_files]
                })
        except Exception as e:
            logger.error(f"Error checking file descriptors: {e}")
        
        # Check for memory leaks
        try:
            memory_info = process.memory_info()
            if memory_info.rss > settings.resources.max_memory_mb * 1024 * 1024:
                # Only include available memory info attributes
                memory_details = {
                    'vms': memory_info.vms / (1024 * 1024),
                    'rss': memory_info.rss / (1024 * 1024)
                }
                
                # Add optional memory attributes if available
                for attr in ['shared', 'text', 'lib', 'data', 'dirty']:
                    if hasattr(memory_info, attr):
                        memory_details[attr] = getattr(memory_info, attr) / (1024 * 1024)
                
                leaks.append({
                    'type': 'memory',
                    'current_mb': memory_info.rss / (1024 * 1024),
                    'limit_mb': settings.resources.max_memory_mb,
                    'details': memory_details
                })
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
        
        # Check for thread leaks
        try:
            thread_count = process.num_threads()
            if thread_count > settings.resources.max_threads:
                thread_info = []
                for thread in process.threads():
                    try:
                        thread_info.append(str(thread.id))  # Use thread ID instead of name
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                leaks.append({
                    'type': 'threads',
                    'current': thread_count,
                    'limit': settings.resources.max_threads,
                    'details': thread_info
                })
        except Exception as e:
            logger.error(f"Error checking thread count: {e}")
        
        return leaks
    
    def _init_db(self) -> None:
        """Initialize resource monitoring database."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            c = conn.cursor()
            c.execute("""
            CREATE TABLE IF NOT EXISTS resource_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                disk_io_read_bytes INTEGER NOT NULL,
                disk_io_write_bytes INTEGER NOT NULL,
                network_bytes_sent INTEGER NOT NULL,
                network_bytes_recv INTEGER NOT NULL
            )
            """)
            
            # Create indexes
            c.execute("CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON resource_stats(timestamp)")
            
            conn.commit()
        finally:
            conn.close()
    
    def _save_stats(self, stats: ResourceStats) -> None:
        """Save resource statistics to database."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO resource_stats (
                    timestamp, cpu_percent, memory_percent,
                    disk_io_read_bytes, disk_io_write_bytes,
                    network_bytes_sent, network_bytes_recv
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stats.timestamp.isoformat(),
                    stats.cpu_percent,
                    stats.memory_percent,
                    stats.disk_io_read_bytes,
                    stats.disk_io_write_bytes,
                    stats.network_bytes_sent,
                    stats.network_bytes_recv
                )
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> 'List[ResourceStats]':
        """Get resource statistics for a time period."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            c = conn.cursor()
            
            query = """
                SELECT * FROM resource_stats
                WHERE 1=1
            """
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            c.execute(query, params)
            return [
                ResourceStats(
                    timestamp=datetime.fromisoformat(row[1]),
                    cpu_percent=row[2],
                    memory_percent=row[3],
                    disk_percent=row[4],
                    network_bytes_sent=row[5],
                    network_bytes_recv=row[6],
                    disk_io_read_bytes=row[7],
                    disk_io_write_bytes=row[8]
                )
                for row in c.fetchall()
            ]
        finally:
            conn.close()
    
    def _get_disk_io_rates(self) -> Tuple[float, float]:
        """Calculate disk I/O rates in bytes per second."""
        current_io = psutil.disk_io_counters()
        current_time = datetime.now()
        time_diff = (current_time - self._last_check).total_seconds()
        
        if time_diff > 0:
            read_rate = (
                current_io.read_bytes - self._last_disk_io.read_bytes
            ) / time_diff
            write_rate = (
                current_io.write_bytes - self._last_disk_io.write_bytes
            ) / time_diff
        else:
            read_rate = write_rate = 0.0
        
        self._last_disk_io = current_io
        return read_rate, write_rate
    
    def _get_network_rates(self) -> Tuple[float, float]:
        """Calculate network I/O rates in bytes per second."""
        current_io = psutil.net_io_counters()
        current_time = datetime.now()
        time_diff = (current_time - self._last_check).total_seconds()
        
        if time_diff > 0:
            send_rate = (
                current_io.bytes_sent - self._last_network_io.bytes_sent
            ) / time_diff
            recv_rate = (
                current_io.bytes_recv - self._last_network_io.bytes_recv
            ) / time_diff
        else:
            send_rate = recv_rate = 0.0
        
        self._last_network_io = current_io
        return send_rate, recv_rate
    
    def check_resources(self) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Check if system resources are within acceptable limits.
        
        Returns:
            Tuple of (can_process, reason_if_not, current_stats)
        """
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Get disk and network I/O rates
            disk_read_rate, disk_write_rate = self._get_disk_io_rates()
            network_send_rate, network_recv_rate = self._get_network_rates()
            
            # Create stats object
            stats = ResourceStats(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_io_read_bytes=int(disk_read_rate),
                disk_io_write_bytes=int(disk_write_rate),
                network_bytes_sent=int(network_send_rate),
                network_bytes_recv=int(network_recv_rate)
            )
            
            # Save stats
            self._save_stats(stats)
            
            # Update last check time
            self._last_check = datetime.now()
            
            # Check resource limits
            resource_status = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_read_rate_mb': disk_read_rate / 1024 / 1024,
                'disk_write_rate_mb': disk_write_rate / 1024 / 1024,
                'network_send_rate_mb': network_send_rate / 1024 / 1024,
                'network_recv_rate_mb': network_recv_rate / 1024 / 1024
            }
            
            # CPU check
            if cpu_percent > settings.resources.max_cpu_percent:
                return False, f"CPU usage too high: {cpu_percent}%", resource_status
            
            # Memory check
            if memory_percent > settings.resources.max_memory_percent:
                return False, f"Memory usage too high: {memory_percent}%", resource_status
            
            # Disk I/O check
            max_disk_mb = settings.resources.max_disk_io_mb_per_sec
            if disk_read_rate / 1024 / 1024 > max_disk_mb:
                return False, f"Disk read rate too high: {disk_read_rate/1024/1024:.1f} MB/s", resource_status
            if disk_write_rate / 1024 / 1024 > max_disk_mb:
                return False, f"Disk write rate too high: {disk_write_rate/1024/1024:.1f} MB/s", resource_status
            
            # Network I/O check
            max_network_mb = settings.resources.max_network_io_mb_per_sec
            if network_send_rate / 1024 / 1024 > max_network_mb:
                return False, f"Network send rate too high: {network_send_rate/1024/1024:.1f} MB/s", resource_status
            if network_recv_rate / 1024 / 1024 > max_network_mb:
                return False, f"Network receive rate too high: {network_recv_rate/1024/1024:.1f} MB/s", resource_status
            
            return True, None, resource_status
            
        except Exception as e:
            error_id = error_store.record_error(e, 'resource_monitor')
            logger.error(f"Error checking resources (ID: {error_id}): {e}")
            return False, f"Error checking resources: {str(e)}", {}
    
    def get_resource_usage_report(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate resource usage report for the specified time period.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Dictionary containing resource usage statistics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        stats = self.get_stats(start_time, end_time)
        if not stats:
            return {
                'error': 'No data available for the specified time period'
            }
        
        # Calculate averages
        cpu_avg = sum(s.cpu_percent for s in stats) / len(stats)
        memory_avg = sum(s.memory_percent for s in stats) / len(stats)
        disk_read_avg = sum(s.disk_io_read_bytes for s in stats) / len(stats)
        disk_write_avg = sum(s.disk_io_write_bytes for s in stats) / len(stats)
        network_send_avg = sum(s.network_bytes_sent for s in stats) / len(stats)
        network_recv_avg = sum(s.network_bytes_recv for s in stats) / len(stats)
        
        # Calculate peaks
        cpu_peak = max(s.cpu_percent for s in stats)
        memory_peak = max(s.memory_percent for s in stats)
        disk_read_peak = max(s.disk_io_read_bytes for s in stats)
        disk_write_peak = max(s.disk_io_write_bytes for s in stats)
        network_send_peak = max(s.network_bytes_sent for s in stats)
        network_recv_peak = max(s.network_bytes_recv for s in stats)
        
        return {
            'period_hours': hours,
            'total_samples': len(stats),
            'averages': {
                'cpu_percent': round(cpu_avg, 1),
                'memory_percent': round(memory_avg, 1),
                'disk_read_mb_per_sec': round(disk_read_avg / 1024 / 1024, 2),
                'disk_write_mb_per_sec': round(disk_write_avg / 1024 / 1024, 2),
                'network_send_mb_per_sec': round(network_send_avg / 1024 / 1024, 2),
                'network_recv_mb_per_sec': round(network_recv_avg / 1024 / 1024, 2)
            },
            'peaks': {
                'cpu_percent': round(cpu_peak, 1),
                'memory_percent': round(memory_peak, 1),
                'disk_read_mb_per_sec': round(disk_read_peak / 1024 / 1024, 2),
                'disk_write_mb_per_sec': round(disk_write_peak / 1024 / 1024, 2),
                'network_send_mb_per_sec': round(network_send_peak / 1024 / 1024, 2),
                'network_recv_mb_per_sec': round(network_recv_peak / 1024 / 1024, 2)
            },
            'current': {
                'cpu_percent': round(stats[0].cpu_percent, 1),
                'memory_percent': round(stats[0].memory_percent, 1),
                'disk_read_mb_per_sec': round(stats[0].disk_io_read_bytes / 1024 / 1024, 2),
                'disk_write_mb_per_sec': round(stats[0].disk_io_write_bytes / 1024 / 1024, 2),
                'network_send_mb_per_sec': round(stats[0].network_bytes_sent / 1024 / 1024, 2),
                'network_recv_mb_per_sec': round(stats[0].network_bytes_recv / 1024 / 1024, 2)
            }
        }