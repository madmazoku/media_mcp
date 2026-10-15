from .wan_latent_nodes import (
    WanLatentPrefixToVideo,
    WanLatentPrefixLock,
    WanLatentPrefixRestore
)

NODE_CLASS_MAPPINGS = {
    "WanLatentPrefixToVideo": WanLatentPrefixToVideo,
    "WanLatentPrefixLock": WanLatentPrefixLock,
    "WanLatentPrefixRestore": WanLatentPrefixRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLatentPrefixToVideo": "Wan 2.2 Latent Prefix To Video (Media-MCP)",
    "WanLatentPrefixLock": "Wan 2.2 Latent Prefix Lock (Media-MCP)",
    "WanLatentPrefixRestore": "Wan 2.2 Latent Prefix Restore (Media-MCP)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
