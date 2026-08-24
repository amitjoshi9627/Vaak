class VaakError(Exception):
    """Base exception for all Vaak errors."""


class ConfigurationError(VaakError):
    """Raised when Vaak configuration is invalid."""


class RegistryError(VaakError):
    """Raised when registry operations fail."""
