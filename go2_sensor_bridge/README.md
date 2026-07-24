# go2_sensor_bridge

Independent ROS 2 boundary for the Go2 auxiliary sensors used by T4.2-R3.
It accepts only standard ROS messages, removes points belonging to the body or
legs, enforces range/height limits, preserves organized LiDAR shape, republishes
front RGB and IMU, and builds the point cloud consumed by Collision Monitor.

The bridge does not deserialize Python objects or consume map/episode truth.
Its JSON audit records rates, timestamps, frame IDs, dropouts, filtering counts,
and the active D435i/LiDAR source contract.
