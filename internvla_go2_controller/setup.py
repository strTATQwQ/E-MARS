from setuptools import find_packages, setup


package_name = "internvla_go2_controller"

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
    maintainer="InternNav T3",
    maintainer_email="noreply@example.invalid",
    description="Continuous Isaac Go2 controller bridge for InternVLA and Nav2.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "internvla_go2_controller_bridge = internvla_go2_controller.bridge_node:main",
        ],
    },
)
