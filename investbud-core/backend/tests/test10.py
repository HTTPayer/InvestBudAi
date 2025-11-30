from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider

node_provider = NodeHTTPProvider(endpoint_uri="https://v6-pegasus-node-02.origin-trail.network:8900", api_version="v1")
blockchain_provider = BlockchainProvider(
    "otp:20430"  # Neuroweb Testnet ID,
)

dkg = DKG(node_provider, blockchain_provider)

print(dkg.node.info)
# if successfully connected, this should print the dictionary with node version
# { "version": "8.X.X" }