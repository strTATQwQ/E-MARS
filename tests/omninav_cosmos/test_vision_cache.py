import pytest

from omninav_cosmos.vision_cache import EpisodeVisionCache, VisionCacheKey


def key(episode, frame, *, width=640, model="model-a", view="front"):
    return VisionCacheKey(episode, frame, view, width, 480, model)


def test_cache_is_a_bounded_ring():
    cache = EpisodeVisionCache(capacity=2)
    cache.begin_episode("ep-a")
    cache.put(key("ep-a", 1), "one")
    cache.put(key("ep-a", 2), "two")
    cache.put(key("ep-a", 3), "three")
    assert cache.get(key("ep-a", 1)) is None
    assert cache.get(key("ep-a", 2)) == "two"
    assert cache.get(key("ep-a", 3)) == "three"


def test_new_episode_hard_clears_and_rejects_old_episode_access():
    cache = EpisodeVisionCache()
    cache.begin_episode("ep-a")
    cache.put(key("ep-a", 1), object())
    cache.begin_episode("ep-b")
    assert cache.keys() == ()
    with pytest.raises(RuntimeError, match="cross-episode"):
        cache.get(key("ep-a", 1))


def test_resolution_or_model_change_invalidates_same_view():
    cache = EpisodeVisionCache()
    cache.begin_episode("ep-a")
    cache.put(key("ep-a", 1), "old")
    cache.put(key("ep-a", 2, width=320), "new-size")
    assert cache.get(key("ep-a", 1)) is None
    cache.put(key("ep-a", 3, width=320, model="model-b"), "new-model")
    assert cache.get(key("ep-a", 2, width=320)) is None


def test_disabled_cache_never_stores_features():
    cache = EpisodeVisionCache(enabled=False)
    cache.begin_episode("ep-a")
    cache.put(key("ep-a", 1), "ignored")
    assert cache.get(key("ep-a", 1)) is None

