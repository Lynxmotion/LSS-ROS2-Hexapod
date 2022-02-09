int8 ABSOLUTE = 0
int8 RELATIVE = 1
int8 DEG_SEC = 2

# robot behavior modes
int8 SITTING = 0
int8 ACTIVE = 1

# set the robot pose state
int8 behavior

# the requested height of the robot off the ground
float32 standing_height

# the robot will start waking for any non-zero value
float32 walking_speed

# how the heading value should be interpreted
int8 heading_mode

# the heading command
float32 heading

# change the orientation of the robot body
float32 roll
float32 pitch

