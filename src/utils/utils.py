import os, ctypes, logging

logger = logging.getLogger(__name__)


def load_cpp_library(lib_name: str, path_to_library: str = "deps/") -> ctypes.CDLL:
    try:
        # Get the absolute path to the library, assuming it's in the same directory
        lib_path = os.path.join(path_to_library, lib_name)
        return ctypes.CDLL(lib_path)
    except OSError as e:
        logger.error(f"Error loading library: {e}")
        logger.error(f"Please make sure '{lib_name}' is compiled and in the deps/ directory that is found at the root of the project.")
        exit()
