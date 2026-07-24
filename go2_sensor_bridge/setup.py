from setuptools import find_packages, setup

package_name = "go2_sensor_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="InternNav T4",
    maintainer_email="noreply@example.invalid",
    description="Standard ROS 2 Go2 multi-sensor bridge with audited self filtering.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "go2_sensor_bridge = go2_sensor_bridge.bridge_node:main",
        ],
    },
)
