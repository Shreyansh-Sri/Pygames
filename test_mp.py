import ctypes
import sys

if sys.platform == "win32":
    print("Fixing free...")
    import ctypes.util
    libc_path = ctypes.util.find_library('ucrtbase.dll')
    if libc_path:
        libc = ctypes.CDLL(libc_path)
        ctypes.cdll.msvcrt.free = libc.free
        print("Fixed!")
    else:
        print("ucrtbase.dll not found")

print("Importing mediapipe...")
import mediapipe as mp
print("MediaPipe imported.")
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_faces=1,
)
print("Creating FaceLandmarker...")
try:
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    print("SUCCESS: Face landmarker created!")
except Exception as e:
    print(f"FAILED: {e}")
