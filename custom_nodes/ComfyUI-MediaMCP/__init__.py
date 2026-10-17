import importlib
from pathlib import Path
from . import wan_latent_nodes

_LAST_MTIME = 0.0


def _ensure_fresh_nodes():
    global _LAST_MTIME
    try:
        p = Path(wan_latent_nodes.__file__)
        mtime = p.stat().st_mtime
        if _LAST_MTIME == 0.0:
            _LAST_MTIME = mtime
        elif mtime > _LAST_MTIME:
            importlib.reload(wan_latent_nodes)
            _LAST_MTIME = mtime
    except Exception:
        pass


class _HotReloadNodeProxy:
    target_cls_name: str = ""

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_fresh_nodes()
        return getattr(wan_latent_nodes, cls.target_cls_name).INPUT_TYPES()

    def execute(self, *args, **kwargs):
        _ensure_fresh_nodes()
        target_cls = getattr(wan_latent_nodes, self.target_cls_name)
        return target_cls().execute(*args, **kwargs)


class WanLatentPrefixToVideo(_HotReloadNodeProxy):
    target_cls_name = "WanLatentPrefixToVideo"
    RETURN_TYPES = wan_latent_nodes.WanLatentPrefixToVideo.RETURN_TYPES
    RETURN_NAMES = wan_latent_nodes.WanLatentPrefixToVideo.RETURN_NAMES
    FUNCTION = "execute"
    CATEGORY = wan_latent_nodes.WanLatentPrefixToVideo.CATEGORY


class WanLatentPrefixLock(_HotReloadNodeProxy):
    target_cls_name = "WanLatentPrefixLock"
    RETURN_TYPES = wan_latent_nodes.WanLatentPrefixLock.RETURN_TYPES
    RETURN_NAMES = wan_latent_nodes.WanLatentPrefixLock.RETURN_NAMES
    FUNCTION = "execute"
    CATEGORY = wan_latent_nodes.WanLatentPrefixLock.CATEGORY


class WanLatentPrefixRestore(_HotReloadNodeProxy):
    target_cls_name = "WanLatentPrefixRestore"
    RETURN_TYPES = wan_latent_nodes.WanLatentPrefixRestore.RETURN_TYPES
    RETURN_NAMES = wan_latent_nodes.WanLatentPrefixRestore.RETURN_NAMES
    FUNCTION = "execute"
    CATEGORY = wan_latent_nodes.WanLatentPrefixRestore.CATEGORY


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
