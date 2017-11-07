# DigiPyRo
Digitally rotates a movie and allows for single-particle tracking. 
Originally designed to intuitively show Coriolis force effects through the appearance of inertial circles when digitally rotating film of a ball oscillating on a parabolic surface.

# Installing
1. `curl https://raw.githubusercontent.com/sam-may/DigiPyRo/master/install.sh > install.sh`
2. `source install.sh`

# Running
1. Check if openCV and DigiPyRo are correctly linked:
```
python
import cv2
```
2. Create a synthetic movie: `python synths.py`
3. Digitally rotate a movie: `python DigiPyRo.py`
