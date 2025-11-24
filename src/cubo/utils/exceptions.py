"""
CUBO Custom Exceptions
Custom exception classes for different error categories in the CUBO system.
"""

from typing import Any, Dict, Optional


class CUBOError(Exception):
    """Base exception class for all CUBO-related errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class DatabaseError(CUBOError):
    """Errors related to database operations (vector stores, file storage, etc.)."""
    pass


class EmbeddingError(CUBOError):
    """Errors related to embedding generation and processing."""
    pass


class FileOperationError(CUBOError):
    """Errors related to file operations (reading, writing, path issues)."""
    pass


class ConfigurationError(CUBOError):
    """Errors related to configuration loading and validation."""
    pass


class ModelLoadError(CUBOError):
    """Errors related to model loading and initialization."""
    pass


class RetrievalError(CUBOError):
    """Errors related to document retrieval operations."""
    pass


class ProcessingError(CUBOError):
    """Errors related to document processing and chunking."""
    pass


class ValidationError(CUBOError):
    """Errors related to input validation."""
    pass


class ServiceError(CUBOError):
    """Errors related to service operations and external dependencies."""
    pass


class HealthCheckError(CUBOError):
    """Errors related to system health monitoring."""
    pass


class GUIError(CUBOError):
    """Errors related to GUI operations."""
    pass


class NetworkError(CUBOError):
    """Errors related to network operations."""
    pass


# Specific error codes for common scenarios
class DocumentAlreadyExistsError(DatabaseError):
    """Raised when attempting to add a document that already exists."""

    def __init__(self, filename: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Document '{filename}' already exists in the database",
            "DOC_EXISTS",
            details
        )


class DocumentNotFoundError(DatabaseError):
    """Raised when a requested document is not found."""

    def __init__(self, document_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Document '{document_id}' not found",
            "DOC_NOT_FOUND",
            details
        )


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""

    def __init__(self, text_preview: str, details: Optional[Dict[str, Any]] = None):
        preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        super().__init__(
            f"Failed to generate embeddings for text: '{preview}'",
            "EMBEDDING_FAILED",
            details
        )


class ModelNotAvailableError(ModelLoadError):
    """Raised when required model is not available."""

    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Model '{model_name}' is not available",
            "MODEL_UNAVAILABLE",
            details
        )


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""

    def __init__(self, config_key: str, value: Any, expected: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Invalid configuration for '{config_key}': got {value}, expected {expected}",
            "INVALID_CONFIG",
            details
        )


class RetrievalMethodUnavailableError(RetrievalError):
    """Raised when requested retrieval method is not available."""

    def __init__(self, method: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Retrieval method '{method}' is not available",
            "METHOD_UNAVAILABLE",
            details
        )


class FileAccessError(FileOperationError):
    """Raised when file access fails."""

    def __init__(self, filepath: str, operation: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Failed to {operation} file '{filepath}'",
            "FILE_ACCESS_ERROR",
            details
        )


class ServiceTimeoutError(ServiceError):
    """Raised when a service operation times out."""

    def __init__(self, service_name: str, timeout_seconds: float, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Service '{service_name}' timed out after {timeout_seconds} seconds",
            "SERVICE_TIMEOUT",
            details
        )


class HealthCheckFailedError(HealthCheckError):
    """Raised when a health check fails."""

    def __init__(self, check_name: str, status: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Health check '{check_name}' failed with status: {status}",
            "HEALTH_CHECK_FAILED",
            details
        )
