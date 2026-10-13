"""Package marker."""
import os

# Windows Media Foundation opens webcams ~10x faster without its hardware transforms
# (2 s instead of ~20 s). Must be set before OpenCV opens a camera.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
