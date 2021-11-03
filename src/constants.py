"""
Class Constants
"""

"""
Associating each key with a finger
Hand encoding:
R = Right
L = Left

Finger encoding:
T = Thumb
I = Index
M = Middle
R = Ring 
P = Pinky
"""

KEY_MAP = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", \
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", \
            "space", "shift", "right shift", "backspace", "'", "-", \
           "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".", "&", "%", "+", "\\"]

FINGERS = ["RT", "RI", "RM", "RR", "RP", "LT", "LI", "LM", "LR", "LP"]

FINGER_MAP = ["LP", "LI", "LM", "LM", "LM", "LI", "LI", "RI", "RM", "RI", \
              "RM", "RR", "RI", "RI", "RR", "RP", "LP", "LI", "LR", "LI", \
              "RI", "LI", "LR", "LR", "RI", "LP", "RT", "LP", "RP", "RP", \
              "RP", "RR", "LP", "LR", "LM", "LI", "LI", "RI", "RI", "RM", \
              "RR", "RR", ".", "RI", "LI", "RP", "RP"]

# Side note: I tend to use RT for 'space' but some may use LT and RT interchangeably
