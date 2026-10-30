import importlib
from pathlib import Path
from . import wan_latent_nodes
from . import ltx_latent_nodes

_LAST_MTIME_WAN = 0.0
_LAST_MTIME_LTX = 0.0


def _ensure_fresh_nodes():
    global _LAST_MTIME_WAN, _LAST_MTIME_LTX
    try:
        p_wan = Path(wan_latent_nodes.__file__)
        mtime_wan = p_wan.stat().st_mtime
        if _LAST_MTIME_WAN == 0.0:
            _LAST_MTIME_WAN = mtime_wan
        elif mtime_wan > _LAST_MTIME_WAN:
            importlib.reload(wan_latent_nodes)
            _LAST_MTIME_WAN = mtime_wan
    except Exception:
        pass

    try:
        p_ltx = Path(ltx_latent_nodes.__file__)
        mtime_ltx = p_ltx.stat().st_mtime
        if _LAST_MTIME_LTX == 0.0:
            _LAST_MTIME_LTX = mtime_ltx
        elif mtime_ltx > _LAST_MTIME_LTX:
            importlib.reload(ltx_latent_nodes)
            _LAST_MTIME_LTX = mtime_ltx
    except Exception:
        pass


class _HotReloadWanNodeProxy:
    target_cls_name: str = ""

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_fresh_nodes()
        return getattr(wan_latent_nodes, cls.target_cls_name).INPUT_TYPES()

    def execute(self, *args, **kwargs):
        _ensure_fresh_nodes()
        target_cls = getattr(wan_latent_nodes, self.target_cls_name)
        return target_cls().execute(*args, **kwargs)


class _HotReloadLtxNodeProxy:
    target_cls_name: str = ""

    @classmethod
    def INPUT_TYPES(cls):
        _ensure_fresh_nodes()
        return getattr(ltx_latent_nodes, cls.target_cls_name).INPUT_TYPES()

    def execute(self, *args, **kwargs):
        _ensure_fresh_nodes()
        target_cls = getattr(ltx_latent_nodes, self.target_cls_name)
        return target_cls().execute(*args, **kwargs)


class WanLatentPrefixToVideo(_HotReloadWanNodeProxy):
    target_cls_name = "WanLatentPrefixToVideo"
    RETURN_TYPES = wan_latent_nodes.WanLatentPrefixToVideo.RETURN_TYPES
    RETURN_NAMES = wan_latent_nodes.WanLatentPrefixToVideo.RETURN_NAMES
    FUNCTION = "execute"
    CATEGORY = wan_latent_nodes.WanLatentPrefixToVideo.CATEGORY


class WanLatentPrefixLock(_HotReloadWanNodeProxy):
    target_cls_name = "WanLatentPrefixLock"
    RETURN_TYPES = wan_latent_nodes.WanLatentPrefixLock.RETURN_TYPES
    RETURN_NAMES = wan_latent_nodes.WanLatentPrefixLock.RETURN_NAMES
    FUNCTION = "execute"
    CATEGORY = wan_latent_nodes.WanLatentPrefixLock.CATEGORY


class WanLatentPrefixRestore(_HotReloadWanNodeProxy):
    target_cls_name = "WanLatentPrefixRestore"
    RETURN_TYPES = wan_latent_nodes.WanLatentPrefixRestore.RETURN_TYPES
    RETURN_NAMES = wan_latent_nodes.WanLatentPrefixRestore.RETURN_NAMES
    FUNCTION = "execute"
    CATEGORY = wan_latent_nodes.WanLatentPrefixRestore.CATEGORY


class LTXLatentPrefixToVideo(_HotReloadLtxNodeProxy):
    target_cls_name = "LTXLatentPrefixToVideo"
    RETURN_TYPES = ltx_latent_nodes.LTXLatentPrefixToVideo.RETURN_TYPES
    RETURN_NAMES = ltx_latent_nodes.LTXLatentPrefixToVideo.RETURN_NAMES
    FUNCTION = "execute"
    CATEGORY = ltx_latent_nodes.LTXLatentPrefixToVideo.CATEGORY


NODE_CLASS_MAPPINGS = {
    "WanLatentPrefixToVideo": WanLatentPrefixToVideo,
    "WanLatentPrefixLock": WanLatentPrefixLock,
    "WanLatentPrefixRestore": WanLatentPrefixRestore,
    "LTXLatentPrefixToVideo": LTXLatentPrefixToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLatentPrefixToVideo": "Wan 2.2 Latent Prefix To Video (Media-MCP)",
    "WanLatentPrefixLock": "Wan 2.2 Latent Prefix Lock (Media-MCP)",
    "WanLatentPrefixRestore": "Wan 2.2 Latent Prefix Restore (Media-MCP)",
    "LTXLatentPrefixToVideo": "LTX-2.3 Latent Prefix To Video (Media-MCP)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
