"""
Application settings module.

Loads and validates application configuration.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
import threading
try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("python-dotenv is required. Please install it with: pip install python-dotenv")

try:
    from pydantic import BaseModel, validator, Field
except ImportError:
    raise ImportError("pydantic is required. Please install it with: pip install pydantic")

logger = logging.getLogger(__name__)

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigurationManager:
    """Manages configuration loading and reloading."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._settings = None
            self._env_file = None
            self._last_load_time = None
            self._initialized = True
    
    def load(self, env_file: Optional[str] = None) -> 'Settings':
        """Load settings from environment file."""
        self._env_file = env_file
        
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # Load environment-specific defaults
        env = os.getenv('ENVIRONMENT', 'development')
        self._load_env_defaults(env)
        
        # Create settings
        self._settings = load_settings()
        self._last_load_time = os.path.getmtime(env_file) if env_file else None
        
        return self._settings
    
    def _load_env_defaults(self, env: str) -> None:
        """Load environment-specific default values."""
        defaults = {
            'development': {
                'DEBUG': 'true',
                'LOG_LEVEL': 'DEBUG',
                'DB_POOL_SIZE': '3',
                'MAX_RETRIES': '5',
                'RESOURCE_CHECK_INTERVAL': '5'
            },
            'testing': {
                'DEBUG': 'true',
                'LOG_LEVEL': 'DEBUG',
                'DB_POOL_SIZE': '2',
                'MAX_RETRIES': '2',
                'RESOURCE_CHECK_INTERVAL': '1'
            },
            'production': {
                'DEBUG': 'false',
                'LOG_LEVEL': 'INFO',
                'DB_POOL_SIZE': '10',
                'MAX_RETRIES': '3',
                'RESOURCE_CHECK_INTERVAL': '1'
            }
        }
        
        # Set defaults if not already set
        for key, value in defaults.get(env, {}).items():
            if key not in os.environ:
                os.environ[key] = value
    
    def reload_if_changed(self) -> Optional['Settings']:
        """Reload settings if environment file has changed."""
        if not self._env_file or not self._last_load_time:
            return None
        
        current_mtime = os.path.getmtime(self._env_file)
        if current_mtime > self._last_load_time:
            logger.info("Configuration file changed, reloading settings")
            return self.load(self._env_file)
        
        return None
    
    def validate_required_settings(self) -> None:
        """Validate that all required settings are present and valid."""
        required_settings = {
            'JINA_API_KEY': 'Jina API key is required',
            'QDRANT_API_KEY': 'Qdrant API key is required',
            'QDRANT_URL': 'Qdrant URL is required',
            'DB_PATH': 'Database path is required'
        }
        
        missing = []
        for key, message in required_settings.items():
            if not os.getenv(key):
                missing.append(message)
        
        if missing:
            raise ConfigValidationError(
                "Missing required settings:\n" + "\n".join(missing)
            )

class DatabaseConfigModel(BaseModel):
    """Database configuration validation model."""
    path: str
    pool_size: int = Field(ge=1, le=100)
    max_overflow: int = Field(ge=0, le=100)
    pool_timeout: int = Field(ge=1, le=300)
    pool_recycle: int = Field(ge=300, le=7200)
    
    @validator('path')
    def validate_path(cls, v):
        if not v:
            raise ValueError("Database path cannot be empty")
        return v

class QdrantConfigModel(BaseModel):
    """Qdrant configuration validation model."""
    host: str
    port: int = Field(ge=1, le=65535)
    collection_name: str
    api_key: Optional[str]
    url: Optional[str]
    cluster_name: Optional[str]
    
    @validator('collection_name')
    def validate_collection_name(cls, v):
        if not v:
            raise ValueError("Collection name cannot be empty")
        return v

class JinaConfigModel(BaseModel):
    """Jina configuration validation model."""
    api_key: str
    embedding_model: str
    segmenter_model: str
    embedding_dimensions: int = Field(ge=1, le=4096)
    batch_size: int = Field(ge=1, le=1000)
    max_tokens_per_batch: int = Field(ge=1000, le=50000)
    late_chunking: bool
    request_timeout: int = Field(ge=1, le=300)
    rate_limit_minute: int = Field(ge=1, le=1000)
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("API key cannot be empty")
        return v

@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800

@dataclass
class QdrantConfig:
    """Qdrant configuration."""
    host: str
    port: int
    collection_name: str
    api_key: Optional[str] = None
    url: Optional[str] = None
    cluster_name: Optional[str] = None

@dataclass
class JinaConfig:
    """Jina configuration."""
    api_key: str
    embedding_model: str = "jina-embeddings-v2-base-en"
    segmenter_model: str = "o200k_base"
    embedding_dimensions: int = 1024
    batch_size: int = 32
    max_tokens_per_batch: int = 8000
    late_chunking: bool = True
    request_timeout: int = 60
    rate_limit_minute: int = 100

@dataclass
class ResourceConfig:
    """Resource limits configuration."""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_disk_io_mb_per_sec: float = 100.0
    max_network_io_mb_per_sec: float = 100.0
    check_interval_seconds: int = 1
    stats_retention_days: int = 7
    max_open_files: int = 1000
    max_threads: int = 100
    max_memory_mb: int = 4096  # 4GB
    cleanup_interval_seconds: int = 300  # 5 minutes
    temp_dir_cleanup_age_hours: int = 24

@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    max_retries: int = 3
    retry_delay_base_seconds: int = 30
    max_retry_delay_seconds: int = 300
    batch_size: int = 50  # Aligned with previous configuration
    max_queue_size: int = 1000
    worker_threads: int = 4
    processing_timeout_seconds: int = 300

@dataclass
class ValidationConfig:
    """Document validation configuration."""
    max_document_size_mb: float
    min_content_length: int
    max_content_length: int
    allowed_file_types: 'List[str]' = field(default_factory=lambda: ['.md', '.txt', '.rst'])

@dataclass
class Settings:
    """Application settings."""
    db: DatabaseConfig
    qdrant: QdrantConfig
    jina: JinaConfig
    resources: ResourceConfig
    processing: ProcessingConfig
    validation: ValidationConfig
    environment: str
    log_level: str
    debug: bool
    db_path: str = "data/documents.db"

def load_settings() -> Settings:
    """
    Load application settings from environment variables.
    
    Returns:
        Settings object containing all configuration
    """
    # Validate required settings
    config_manager = ConfigurationManager()
    config_manager.validate_required_settings()
    
    # Load and validate all settings
    try:
        # Database configuration
        db_config = DatabaseConfig(
            path=os.getenv('DB_PATH', 'documents.db'),
            pool_size=int(os.getenv('DB_POOL_SIZE', '5')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '10')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '1800'))
        )
        DatabaseConfigModel(**asdict(db_config))
        
        # Qdrant configuration
        qdrant_config = QdrantConfig(
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', '6333')),
            collection_name=os.getenv('QDRANT_COLLECTION_NAME', 'documents'),
            api_key=os.getenv('QDRANT_API_KEY'),
            url=os.getenv('QDRANT_URL'),
            cluster_name=os.getenv('QDRANT_CLUSTER_NAME')
        )
        QdrantConfigModel(**asdict(qdrant_config))
        
        # Jina configuration
        jina_config = JinaConfig(
            api_key=os.getenv('JINA_API_KEY', ''),
            embedding_model=os.getenv('JINA_EMBEDDING_MODEL', 'jina-embeddings-v2-base-en'),
            segmenter_model=os.getenv('JINA_SEGMENTER_MODEL', 'o200k_base'),
            embedding_dimensions=int(os.getenv('EMBEDDING_DIMENSIONS', '1024')),
            batch_size=int(os.getenv('JINA_BATCH_SIZE', '32')),
            max_tokens_per_batch=int(os.getenv('MAX_TOKENS_PER_BATCH', '8000')),
            late_chunking=os.getenv('LATE_CHUNKING', 'true').lower() == 'true',
            request_timeout=int(os.getenv('JINA_REQUEST_TIMEOUT', '60')),
            rate_limit_minute=int(os.getenv('JINA_RATE_LIMIT_MINUTE', '100'))
        )
        JinaConfigModel(**asdict(jina_config))
        
        # Resource configuration
        resource_config = ResourceConfig(
            max_cpu_percent=float(os.getenv('MAX_CPU_PERCENT', '80.0')),
            max_memory_percent=float(os.getenv('MAX_MEMORY_PERCENT', '80.0')),
            max_disk_io_mb_per_sec=float(os.getenv('MAX_DISK_IO_MB', '100.0')),
            max_network_io_mb_per_sec=float(os.getenv('MAX_NETWORK_IO_MB', '100.0')),
            check_interval_seconds=int(os.getenv('RESOURCE_CHECK_INTERVAL', '1')),
            stats_retention_days=int(os.getenv('STATS_RETENTION_DAYS', '7')),
            max_open_files=int(os.getenv('MAX_OPEN_FILES', '1000')),
            max_threads=int(os.getenv('MAX_THREADS', '100')),
            max_memory_mb=int(os.getenv('MAX_MEMORY_MB', '4096')),
            cleanup_interval_seconds=int(os.getenv('CLEANUP_INTERVAL_SECONDS', '300')),
            temp_dir_cleanup_age_hours=int(os.getenv('TEMP_DIR_CLEANUP_AGE_HOURS', '24'))
        )
        
        # Processing configuration
        processing_config = ProcessingConfig(
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            retry_delay_base_seconds=int(os.getenv('RETRY_DELAY_BASE', '30')),
            max_retry_delay_seconds=int(os.getenv('MAX_RETRY_DELAY', '300')),
            batch_size=int(os.getenv('BATCH_SIZE', '50')),  # Using BATCH_SIZE for consistency
            max_queue_size=int(os.getenv('MAX_QUEUE_SIZE', '1000')),
            worker_threads=int(os.getenv('WORKER_THREADS', '4')),
            processing_timeout_seconds=int(os.getenv('PROCESSING_TIMEOUT', '300'))
        )
        
        # Validation configuration
        validation_config = ValidationConfig(
            max_document_size_mb=float(os.getenv('MAX_DOCUMENT_SIZE_MB', '10.0')),
            allowed_file_types=os.getenv('ALLOWED_FILE_TYPES', '.md,.txt,.rst').split(','),
            min_content_length=int(os.getenv('MIN_CONTENT_LENGTH', '50')),
            max_content_length=int(os.getenv('MAX_CONTENT_LENGTH', '1000000'))
        )
        
        settings = Settings(
            db=db_config,
            qdrant=qdrant_config,
            jina=jina_config,
            resources=resource_config,
            processing=processing_config,
            validation=validation_config,
            environment=os.getenv('ENVIRONMENT', 'development'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            db_path=os.getenv('DB_PATH', 'documents.db')
        )
        
        return settings
        
    except Exception as e:
        raise ConfigValidationError(f"Configuration validation failed: {str(e)}")

# Global configuration manager
config_manager = ConfigurationManager()

# Global settings instance
settings = config_manager.load() 