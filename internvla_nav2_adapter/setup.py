from setuptools import find_packages, setup


package_name = "internvla_nav2_adapter"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="InternNav T1",
    maintainer_email="maintainer@example.invalid",
    description="Nav2 adapter for typed InternVLA navigation commands.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "internvla_nav2_shadow = internvla_nav2_adapter.shadow_node:main",
            "internvla_nav2_active = internvla_nav2_adapter.active_node:main",
        ],
    },
)
