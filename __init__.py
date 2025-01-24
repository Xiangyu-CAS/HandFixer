from .nodes import MediapipeHandNode

NODE_CLASS_MAPPINGS = {
    "MediapipeHandNode": MediapipeHandNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediapipeHandNode": "MediapipeHandNode",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']