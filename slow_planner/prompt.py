from __future__ import annotations

from .base import SlowPlannerRequest


SYSTEM_PROMPT = (
    "You are the slow semantic planner in a navigation system. "
    "The fast policy and safety controller execute motion. "
    "Use only the ordered RGB observations, instruction, candidate frontier IDs, pose, and compact history provided. "
    "Never invent a frontier ID or reveal chain-of-thought."
)


def build_prompt(request: SlowPlannerRequest, *, correction: str = "") -> str:
    views = "\n".join(
        f"- image_index={index}, view_id={image.view_id}, pose={list(image.pose)}"
        for index, image in enumerate(request.ordered_images)
    )
    frontiers = "\n".join(
        "- id={id}, relative_xz=[{x:.3f},{z:.3f}], distance_m={distance:.3f}, bearing_deg={bearing:.2f}".format(
            id=item.frontier_id,
            x=item.relative_xz[0],
            z=item.relative_xz[1],
            distance=item.distance_m,
            bearing=item.bearing_deg,
        )
        for item in request.candidate_frontiers
    ) or "- none"
    visited = ",".join(str(value) for value in request.visited_frontiers) or "none"
    history = "\n".join(f"- {item}" for item in request.compact_history[-8:]) or "- none"
    correction_block = f"\nCORRECTION:\n{correction}\n" if correction else ""
    return f"""episode_id: {request.episode_id}
snapshot_id: {request.snapshot_id}
instruction: {request.instruction}
agent_pose: {list(request.agent_pose)}

ORDERED_IMAGES (the attached images use exactly this order):
{views}

CANDIDATE_FRONTIERS:
{frontiers}

VISITED_FRONTIER_IDS: {visited}
COMPACT_HISTORY:
{history}
{correction_block}
Choose exactly one action:
1. select_frontier: use only a listed integer frontier_id; target_relative_xz must be null.
2. target_found: only if the instruction's destination is visibly reached; frontier_id must be null and target_relative_xz must be [x,z].
3. abstain: only when no valid choice exists; both optional fields must be null.

Return only compact JSON with exactly these four keys and no prose:
{{"decision":"select_frontier|target_found|abstain","frontier_id":null,"target_relative_xz":null,"confidence":0.0}}"""
