import numpy as np
gp=np.array([[1,1],[1,1]])
# print(gp)
print(np.pad(gp, (2,0), mode='constant',constant_values=0))