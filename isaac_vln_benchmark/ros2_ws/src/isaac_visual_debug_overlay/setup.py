from glob import glob
from setuptools import find_packages, setup

package_name = "isaac_visual_debug_overlay"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="song",
    maintainer_email="song@example.local",
    description="Visual debug overlay artifacts for Isaac VLN benchmark episodes.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "visual_overlay_node = isaac_visual_debug_overlay.visual_overlay_node:main",
            "video_recorder_node = isaac_visual_debug_overlay.video_recorder_node:main",
        ],
    },
)
