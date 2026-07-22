from setuptools import find_packages, setup


package_name = "internvla_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="InternNav T1",
    maintainer_email="maintainer@example.invalid",
    description="Typed ROS 2 model and client nodes for InternVLA navigation.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "internvla_nav2_oracle_bridge = internvla_ros2.oracle_node:main",
            "internvla_model_node = internvla_ros2.model_node:main",
            "internvla_client_node = internvla_ros2.client_node:main",
            "internvla_replay_runner = internvla_ros2.replay_runner:main",
            "internvla_fault_runner = internvla_ros2.fault_runner:main",
        ],
    },
)
