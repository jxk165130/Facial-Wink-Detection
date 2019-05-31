import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Water.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
