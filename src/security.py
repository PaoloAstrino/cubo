import os
import hashlib
import secrets
from cryptography.fernet import Fernet
from src.logger import logger

class SecurityManager:
    """Security utilities for CUBO: encryption, auditing, and secret management."""

    def __init__(self):
        self.encryption_key = self._get_or_create_key()

    def _get_or_create_key(self) -> bytes:
        """Get encryption key from environment or create one."""
        key_env = os.getenv('CUBO_ENCRYPTION_KEY')
        if key_env:
            # Ensure it's 32 bytes for Fernet
            key = key_env.encode()
            if len(key) != 32:
                logger.warning("CUBO_ENCRYPTION_KEY is not 32 bytes. Hashing to derive a 32-byte key.")
                key = hashlib.sha256(key).digest()
            return key
        else:
            error_msg = "CUBO_ENCRYPTION_KEY environment variable not set. Encryption/decryption will not work. Please set a secure, persistent key."
            logger.critical(error_msg)
            raise ValueError(error_msg)

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        try:
            f = Fernet(self.encryption_key)
            encrypted = f.encrypt(data.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            f = Fernet(self.encryption_key)
            decrypted = f.decrypt(encrypted_data.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data for storage/logging."""
        return hashlib.sha256(data.encode()).hexdigest()

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
        required_vars = ['CUBO_ENCRYPTION_KEY']  # Add more as needed
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
        sanitized = input_str.replace('\n', ' ').replace('\r', ' ')
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        return sanitized.strip()

# Global security manager instance
security_manager = SecurityManager()