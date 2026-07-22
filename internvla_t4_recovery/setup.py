from setuptools import find_packages, setup


package_name = "internvla_t4_recovery"

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
    maintainer="InternNav T4",
    maintainer_email="noreply@example.invalid",
    description="Stuck detection, cache invalidation, and bounded Nav2 recovery.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "internvla_t4_model = internvla_t4_recovery.model_node:main",
            "internvla_t4_adapter = internvla_t4_recovery.adapter_dispatch:main",
            "internvla_t4_recovery = internvla_t4_recovery.recovery_node:main",
        ],
    },
)
