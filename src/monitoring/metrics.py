"""
Monitoring and Metrics Module

This module provides comprehensive logging and monitoring for:
- API calls and performance
- Document processing statistics
- Error tracking and analysis
- System health metrics
"""

import logging
import time
from datetime import datetime
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict
from pathlib import Path

@dataclass
class APIMetrics:
    """Metrics for API calls."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency: float = 0.0
    max_latency: float = 0.0
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        return self.total_latency / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0

@dataclass
class ProcessingMetrics:
    """Metrics for document processing."""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    processing_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return (
            self.processed_documents / self.total_documents 
            if self.total_documents > 0 else 0.0
        )
    
    @property
    def average_chunks_per_doc(self) -> float:
        """Calculate average chunks per document."""
        return (
            self.total_chunks / self.processed_documents 
            if self.processed_documents > 0 else 0.0
        )

class MetricsCollector:
    """Collect and manage metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.api_metrics = defaultdict(APIMetrics)
        self.processing_metrics = ProcessingMetrics()
        self._lock = threading.Lock()
        self.logger = logging.getLogger('metrics')
        
        # Ensure logs directory exists
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        
        # Add metrics file handler if not already added
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            metrics_handler = logging.FileHandler(log_dir / 'metrics.log', mode='a')
            metrics_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(metrics_handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False  # Prevent duplicate logging
    
    def log_api_call(self, 
                     operation: str,
                     start_time: float,
                     success: bool,
                     details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an API call with metrics.
        
        Args:
            operation: Name of the API operation
            start_time: Start time of the call
            success: Whether the call was successful
            details: Additional details about the call
        """
        with self._lock:
            latency = time.time() - start_time
            metrics = self.api_metrics[operation]
            
            metrics.total_calls += 1
            metrics.total_latency += latency
            metrics.max_latency = max(metrics.max_latency, latency)
            
            if success:
                metrics.successful_calls += 1
            else:
                metrics.failed_calls += 1
            
            self.logger.info(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'type': 'api_call',
                'operation': operation,
                'success': success,
                'latency': latency,
                'details': details or {},
                'metrics': asdict(metrics)
            }))
    
    def log_document_processing(self,
                              success: bool,
                              chunks_created: int = 0,
                              embeddings_generated: int = 0,
                              processing_time: float = 0.0,
                              details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log document processing metrics.
        
        Args:
            success: Whether processing was successful
            chunks_created: Number of chunks created
            embeddings_generated: Number of embeddings generated
            processing_time: Time taken to process
            details: Additional processing details
        """
        with self._lock:
            self.processing_metrics.total_documents += 1
            
            if success:
                self.processing_metrics.processed_documents += 1
                self.processing_metrics.total_chunks += chunks_created
                self.processing_metrics.total_embeddings += embeddings_generated
                self.processing_metrics.processing_time += processing_time
            else:
                self.processing_metrics.failed_documents += 1
            
            self.logger.info(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'type': 'document_processing',
                'success': success,
                'chunks_created': chunks_created,
                'embeddings_generated': embeddings_generated,
                'processing_time': processing_time,
                'details': details or {},
                'metrics': asdict(self.processing_metrics)
            }))
    
    def get_api_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get API metrics for specific operation or all operations."""
        with self._lock:
            if operation:
                return asdict(self.api_metrics[operation])
            return {
                op: asdict(metrics)
                for op, metrics in self.api_metrics.items()
            }
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get document processing metrics."""
        with self._lock:
            return asdict(self.processing_metrics)
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.api_metrics.clear()
            self.processing_metrics = ProcessingMetrics()
            self.logger.info(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'type': 'metrics_reset',
                'message': 'All metrics have been reset'
            }))

# Create a singleton instance
metrics_collector = MetricsCollector()

def log_api_call(operation: str,
                 start_time: float,
                 success: bool,
                 details: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function for logging API calls."""
    metrics_collector.log_api_call(operation, start_time, success, details)

def log_document_processing(success: bool,
                          chunks_created: int = 0,
                          embeddings_generated: int = 0,
                          processing_time: float = 0.0,
                          details: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function for logging document processing."""
    metrics_collector.log_document_processing(
        success, chunks_created, embeddings_generated, processing_time, details
    )

def get_api_metrics(operation: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for getting API metrics."""
    return metrics_collector.get_api_metrics(operation)

def get_processing_metrics() -> Dict[str, Any]:
    """Convenience function for getting processing metrics."""
    return metrics_collector.get_processing_metrics()

def reset_metrics() -> None:
    """Convenience function for resetting metrics."""
    metrics_collector.reset_metrics() 