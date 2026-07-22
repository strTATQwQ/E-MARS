from glob import glob

from setuptools import find_packages, setup


package_name = "internvla_t4_sensors"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="InternNav T4",
    maintainer_email="noreply@example.invalid",
    description="Calibrated Isaac sensor bridge and truth-isolation audit for T4.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "internvla_t4_sensor_bridge = internvla_t4_sensors.sensor_bridge_node:main",
            "internvla_t4_client = internvla_t4_sensors.client_dispatch:main",
            "internvla_t4_nvblox_supervisor = internvla_t4_sensors.nvblox_supervisor_node:main",
            "internvla_t5_nvblox_supervisor = internvla_t4_sensors.t5_nvblox_supervisor_node:main",
            "internvla_t4_odometry_supervisor = internvla_t4_sensors.odometry_supervisor_node:main",
            "internvla_t4_costmap_stage_tracer = internvla_t4_sensors.costmap_stage_tracer_node:main",
            "internvla_real_go2_mission_gateway = internvla_t4_sensors.real_go2_mission_gateway:main",
        ],
    },
)
