#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = [1.75, 1.86, 1.5, 1.64]
y = [69, 72, 52,  60]

plt.xlim(0, len(y)-1)
plt.xlabel("Height (in)")
plt.ylabel("Weight (lbs)")
plt.title("Men's Height vs Weight")
plt.scatter(x, y, color="magenta")
plt.show()
