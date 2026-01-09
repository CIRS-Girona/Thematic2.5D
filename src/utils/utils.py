import os, ctypes

def load_cpp_library(lib_name: str, path_to_library: str = "deps/") -> ctypes.CDLL:
    try:
        # Get the absolute path to the library, assuming it's in the same directory
        lib_path = os.path.join(path_to_library, lib_name)
        return ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"Error loading library: {e}")
        print(f"Please make sure '{lib_name}' is compiled and in the deps/ directory that is found at the root of the project.")
        exit()
