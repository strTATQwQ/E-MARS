from glob import glob
from setuptools import find_packages, setup

package_name = "isaac_vln_benchmark"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/configs", glob("../../../configs/*.yaml")),
        ("share/" + package_name + "/scripts", glob("../../../scripts/*.py") + glob("../../../scripts/*.sh")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="song",
    maintainer_email="song@example.local",
    description="Isaac benchmark for OmniNav plus Step scheduling strategies.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "episode_manager_node = isaac_vln_benchmark.episode_manager_node:main",
            "scene_randomizer_node = isaac_vln_benchmark.scene_randomizer_node:main",
            "goal_oracle_node = isaac_vln_benchmark.goal_oracle_node:main",
            "success_judge_node = isaac_vln_benchmark.success_judge_node:main",
            "delay_injector_node = isaac_vln_benchmark.delay_injector_node:main",
            "reset_on_decision_node = isaac_vln_benchmark.reset_on_decision_node:main",
            "frame_sampler_node = isaac_vln_benchmark.frame_sampler_node:main",
            "semantic_oracle_node = isaac_vln_benchmark.semantic_oracle_node:main",
            "obstacle_controller_node = isaac_vln_benchmark.obstacle_controller_node:main",
            "benchmark_logger_node = isaac_vln_benchmark.benchmark_logger_node:main",
            "go2_benchmark_adapter_node = isaac_vln_benchmark.go2_benchmark_adapter_node:main",
            "grounded_sam_perception_node = isaac_vln_benchmark.grounded_sam_perception_node:main",
            "forced_semantic_oracle_node = isaac_vln_benchmark.forced_semantic_oracle_node:main",
            "public_semantic_heuristic_node = isaac_vln_benchmark.public_semantic_heuristic_node:main",
            "semantic_marker_judge_node = isaac_vln_benchmark.semantic_marker_judge_node:main",
        ],
    },
)
