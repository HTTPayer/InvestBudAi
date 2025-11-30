"""
Supported networks configuration for Alchemy Transfers API.

Reference: https://www.alchemy.com/docs/data/transfers-api/transfers-endpoints/alchemy-get-asset-transfers
"""

from typing import Dict, List, Optional, TypedDict


class NetworkConfig(TypedDict):
    name: str
    chain_id: int
    internal_transfers: bool


# All mainnet networks supported by Alchemy Transfers API
SUPPORTED_NETWORKS: Dict[str, NetworkConfig] = {
    # Tier 1: Major chains (internal transfers on ETH/Polygon only)
    "eth-mainnet": {"name": "Ethereum", "chain_id": 1, "internal_transfers": True},
    "polygon-mainnet": {"name": "Polygon PoS", "chain_id": 137, "internal_transfers": True},
    "arb-mainnet": {"name": "Arbitrum", "chain_id": 42161, "internal_transfers": False},
    "opt-mainnet": {"name": "OP Mainnet", "chain_id": 10, "internal_transfers": False},
    "base-mainnet": {"name": "Base", "chain_id": 8453, "internal_transfers": False},

    # Tier 2: Established L2s/chains
    "arbnova-mainnet": {"name": "Arbitrum Nova", "chain_id": 42170, "internal_transfers": False},
    "zksync-mainnet": {"name": "ZKsync", "chain_id": 324, "internal_transfers": False},
    "linea-mainnet": {"name": "Linea", "chain_id": 59144, "internal_transfers": False},
    "scroll-mainnet": {"name": "Scroll", "chain_id": 534352, "internal_transfers": False},
    "blast-mainnet": {"name": "Blast", "chain_id": 81457, "internal_transfers": False},
    "zora-mainnet": {"name": "Zora", "chain_id": 7777777, "internal_transfers": False},

    # Tier 3: Alt L1s
    "bnb-mainnet": {"name": "BNB Smart Chain", "chain_id": 56, "internal_transfers": False},
    "avax-mainnet": {"name": "Avalanche", "chain_id": 43114, "internal_transfers": False},
    "gnosis-mainnet": {"name": "Gnosis", "chain_id": 100, "internal_transfers": False},
    "celo-mainnet": {"name": "Celo", "chain_id": 42220, "internal_transfers": False},
    "berachain-mainnet": {"name": "Berachain", "chain_id": 80094, "internal_transfers": False},

    # Tier 4: Newer/niche chains
    "worldchain-mainnet": {"name": "World Chain", "chain_id": 480, "internal_transfers": False},
    "shape-mainnet": {"name": "Shape", "chain_id": 360, "internal_transfers": False},
    "unichain-mainnet": {"name": "Unichain", "chain_id": 130, "internal_transfers": False},
    "apechain-mainnet": {"name": "ApeChain", "chain_id": 33139, "internal_transfers": False},
    "soneium-mainnet": {"name": "Soneium", "chain_id": 1868, "internal_transfers": False},
    "ronin-mainnet": {"name": "Ronin", "chain_id": 2020, "internal_transfers": False},
    "story-mainnet": {"name": "Story", "chain_id": 1516, "internal_transfers": False},
    "lens-mainnet": {"name": "Lens", "chain_id": 232, "internal_transfers": False},
    "ink-mainnet": {"name": "Ink", "chain_id": 57073, "internal_transfers": False},
    "anime-mainnet": {"name": "Anime", "chain_id": 69000, "internal_transfers": False},
    "abstract-mainnet": {"name": "Abstract", "chain_id": 2741, "internal_transfers": False},
    "rootstock-mainnet": {"name": "Rootstock", "chain_id": 30, "internal_transfers": False},
    "settlus-mainnet": {"name": "Settlus", "chain_id": 5371, "internal_transfers": False},
    "monad-mainnet": {"name": "Monad", "chain_id": 10143, "internal_transfers": False},
}

# Default network
DEFAULT_NETWORK = "eth-mainnet"


def get_supported_network_ids() -> List[str]:
    """Get list of all supported network IDs."""
    return list(SUPPORTED_NETWORKS.keys())


def is_network_supported(network: str) -> bool:
    """Check if a network is supported."""
    return network in SUPPORTED_NETWORKS


def get_network_config(network: str) -> NetworkConfig:
    """Get configuration for a network. Raises KeyError if not supported."""
    if network not in SUPPORTED_NETWORKS:
        raise KeyError(f"Unsupported network: {network}. Supported: {get_supported_network_ids()}")
    return SUPPORTED_NETWORKS[network]


def supports_internal_transfers(network: str) -> bool:
    """Check if a network supports internal transfer tracking."""
    config = SUPPORTED_NETWORKS.get(network)
    return config["internal_transfers"] if config else False


def get_transfer_categories(network: str) -> List[str]:
    """Get the appropriate transfer categories for a network."""
    base_categories = ["erc20", "external"]
    if supports_internal_transfers(network):
        base_categories.append("internal")
    return base_categories


# Build reverse lookup: chain_id -> network_id
_CHAIN_ID_TO_NETWORK: Dict[int, str] = {
    config["chain_id"]: network_id
    for network_id, config in SUPPORTED_NETWORKS.items()
}


def get_network_by_chain_id(chain_id: int) -> str:
    """
    Get network ID from chain ID.

    Args:
        chain_id: The chain ID (e.g., 1 for Ethereum, 137 for Polygon)

    Returns:
        Network ID string (e.g., 'eth-mainnet', 'polygon-mainnet')

    Raises:
        KeyError if chain_id is not supported
    """
    if chain_id not in _CHAIN_ID_TO_NETWORK:
        raise KeyError(f"Unsupported chain_id: {chain_id}. Supported chain IDs: {list(_CHAIN_ID_TO_NETWORK.keys())}")
    return _CHAIN_ID_TO_NETWORK[chain_id]


def resolve_network(network: Optional[str] = None, chain_id: Optional[int] = None) -> str:
    """
    Resolve network from either network string or chain_id.

    Priority: chain_id > network > default

    Args:
        network: Network ID string (e.g., 'eth-mainnet')
        chain_id: Chain ID integer (e.g., 1)

    Returns:
        Resolved network ID string

    Raises:
        ValueError if neither is valid or both are invalid
    """
    if chain_id is not None:
        return get_network_by_chain_id(chain_id)
    if network is not None:
        if not is_network_supported(network):
            raise ValueError(f"Unsupported network: {network}. Supported: {get_supported_network_ids()}")
        return network
    return DEFAULT_NETWORK
