from .wan_latent_nodes import WanLatentPrefixToVideo

NODE_CLASS_MAPPINGS = {
    "WanLatentPrefixToVideo": WanLatentPrefixToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLatentPrefixToVideo": "Wan 2.2 Latent Prefix To Video (Media-MCP)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
