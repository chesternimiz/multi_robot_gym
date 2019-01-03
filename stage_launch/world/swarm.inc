define kinect ranger
(
  sensor
  (
    range_max 10.0
    fov 58.0
    samples 640
  )
  # generic model properties
  color "black"
  size [ 0.06 0.15 0.03 ]
)

define swarm position
(
  pose [ 0.0 0.0 0.0 0.0 ]

  # odom_error [0.03 0.03 999999 999999 999999 0.02]

  size [ 0.2552 0.2552 0.40 ]
  origin [ 0.0 0.0 0.0 0.0 ]
  gui_nose 1
  drive "omni"
  color "gray"
  
  kinect(pose [ -0.0 0.0 -0.0 0.0 ])
)
