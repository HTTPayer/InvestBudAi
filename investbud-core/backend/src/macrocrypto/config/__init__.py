"""Configuration module for MacroCrypto."""

from .networks import (
    SUPPORTED_NETWORKS,
    DEFAULT_NETWORK,
    NetworkConfig,
    get_supported_network_ids,
    is_network_supported,
    get_network_config,
    supports_internal_transfers,
    get_transfer_categories,
    get_network_by_chain_id,
    resolve_network,
)

__all__ = [
    "SUPPORTED_NETWORKS",
    "DEFAULT_NETWORK",
    "NetworkConfig",
    "get_supported_network_ids",
    "is_network_supported",
    "get_network_config",
    "supports_internal_transfers",
    "get_transfer_categories",
    "get_network_by_chain_id",
    "resolve_network",
]
