#
#  LSS Hexapod
#


# Local Leg Coordinate System
-In the sample walker code (python) we generate coordinates that are local to each leg. These coordinates are relative to the hip servo so that
local coordinates are consistent between all 6 legs. The kinematic system will then do the conversion of global-

# Useful Leg coordinates:

## Neutral pose position relative to each leg's hip servo (Jx1-ST1):
  Here the ZY plane is parallel to the floor, and X is up and down.
   X: 0.005      R: -1.5653
   Y: 0.1577     P: -0.0043
   Z: 0.0        Y: 0.6601


## Neutral pose position relative to base for each leg:
  Here the XY plane is parallel to the floor, and Z is up and down.
  These examples are from eye-balling the positions by hand.

   left-front-foot:
   X:  0.1286        R:  0.0
   Y: -0.1742        P:  0.9596
   Z: -0.0431        Y: -0.0908
  
   left-middle-foot:
   X:  0.212         R:  0.0
   Y:  0.0028        P:  0.9160
   Z: -0.0411        Y:  0.0122

   left-back-foot:
   X:  0.1539        R:  0.0
   Y:  0.1982        P:  0.8898
   Z: -0.0411        Y:  0.8556

   right-front-foot:
   X: -0.1241        R:  0.0
   Y: -0.1822        P:  0.9718
   Z: -0.0436        Y: -2.1796

   right-middle-foot:
   X: -0.2103        R:  0.0
   Y: -0.0035        P:  0.8863
   Z: -0.0383        Y: -3.1242

   right-back-foot:
   X: -0.1376        R: 0.0
   Y:  0.2116        P: 0.8706
   Z: -0.0395        Y: 2.1236

   


Intro
Show base manipulation
   - enable manipulators
   - manipulator translation in X,Y
   - no IMU atm, ground truth
Show leg manipulation
Show Trajectory
   - turn off execution and show preview
   - turn on execution and show actual
Capture a Trajectory

