import hashlib
import os
import secrets
import warnings

try:
    from cryptography.fernet import Fernet

    _HAS_FERNET = True
except Exception:
    Fernet = None
    _HAS_FERNET = False
    warnings.warn("cryptography not available; encryption functions disabled", ImportWarning)

from cubo.config import config
from cubo.utils.logger import logger


class SecurityManager:
    """Security utilities for CUBO: encryption, auditing, and secret management."""

    def __init__(self):
        self._encryption_key = None
        # Optional per-instance config override used in tests (dict)
        self.config = None

    def _get_encryption_key(self) -> bytes:
        """Get encryption key, initializing it lazily."""
        if self._encryption_key is None:
            self._encryption_key = self._get_or_create_key()
        return self._encryption_key

    def _get_or_create_key(self) -> bytes:
        """Get encryption key from environment or create one."""
        key_env = os.getenv("CUBO_ENCRYPTION_KEY")
        if key_env:
            # Fernet keys should be 32 bytes, base64 encoded (44 characters)
            key_str = key_env.encode()
            if len(key_str) == 44:
                # Assume it's a proper base64-encoded Fernet key
                try:
                    # Validate by attempting to create Fernet instance
                    if _HAS_FERNET:
                        Fernet(key_str)
                        return key_str
                    else:
                        warnings.warn(
                            "cryptography is not installed; cannot validate encryption key",
                            ImportWarning,
                        )
                        return hashlib.sha256(key_str).digest()
                except Exception:
                    logger.warning(
                        "CUBO_ENCRYPTION_KEY appears to be base64 but is invalid. Hashing to derive a 32-byte key."
                    )
                    return hashlib.sha256(key_str).digest()
            else:
                logger.warning(
                    "CUBO_ENCRYPTION_KEY is not 44 bytes (base64 encoded). Hashing to derive a 32-byte key."
                )
                return hashlib.sha256(key_str).digest()
        else:
            error_msg = (
                "CUBO_ENCRYPTION_KEY environment variable not set. "
                "Encryption/decryption will not work. Please set a secure, persistent key."
            )
            logger.critical(error_msg)
            # If cryptography isn't available, still allow the app to run without encryption
            if not _HAS_FERNET:
                warnings.warn(
                    "cryptography not installed; encryption features will be disabled",
                    ImportWarning,
                )
                return b""
            raise ValueError(error_msg)

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not _HAS_FERNET:
            raise RuntimeError("cryptography package not installed; encryption not available")
        try:
            f = Fernet(self._get_encryption_key())
            encrypted = f.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not _HAS_FERNET:
            raise RuntimeError("cryptography package not installed; decryption not available")
        try:
            f = Fernet(self._get_encryption_key())
            decrypted = f.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data for storage/logging."""
        return hashlib.sha256(data.encode()).hexdigest()

    def scrub(self, data: str) -> str:
        """Return either the raw data or a hashed representation based on config 'scrub_queries'."""
        if not isinstance(data, str):
            return data
        # Allow per-instance override for testing: security_manager.config in tests
        cfg = getattr(self, "config", None) or config
        # Try both nested and dotted config lookups
        scrub_flag = False
        try:
            if isinstance(cfg, dict):
                # Prefer nested logging key first
                logging_cfg = cfg.get("logging", {})
                scrub_flag = bool(logging_cfg.get("scrub_queries", cfg.get("scrub_queries", False)))
            else:
                scrub_flag = cfg.get("logging.scrub_queries", cfg.get("scrub_queries", False))
        except Exception:
            scrub_flag = False
        if scrub_flag:
            return self.hash_sensitive_data(data)
        return data

    # Backwards-compat wrapper
    def scrub_query(self, data: str) -> str:
        """Legacy wrapper used in older tests. Delegates to `scrub` and returns hashed value when configured.

        Kept for backwards compatibility with tests that expect security_manager.scrub_query.
        """
        return self.scrub(data)

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def audit_log(action: str, user: str = "system", details: dict = None):
        """Log security-relevant actions for auditing."""
        details_str = f" - {details}" if details else ""
        logger.info(f"AUDIT: {action} by {user}{details_str}")

    @staticmethod
    def validate_environment():
        """Validate that required environment variables are set."""
        required_vars = ["CUBO_ENCRYPTION_KEY"]  # Add more as needed
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            logger.warning(f"Missing environment variables: {missing}")
            return False
        return True

    @staticmethod
    def sanitize_input(input_str: str, max_length: int = 1000) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        # Remove potentially dangerous characters
        sanitized = input_str.replace("\n", " ").replace("\r", " ")
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        return sanitized.strip()


# Global security manager instance - initialized without requiring encryption key
security_manager = SecurityManager()
