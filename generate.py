import random
import numpy as np

DELTA_T = 0.032

R_POS_STD = 0.75
R_POS_SCALE = 0.01250
R_POS_VAR  = (R_POS_STD * R_POS_SCALE) **2
R_VEL_VAR  = R_POS_VAR / (DELTA_T **2)
R_REALATION_VAR  = R_POS_VAR / (DELTA_T)
print(f'{R_REALATION_VAR=}')


print(R_POS_STD,
R_POS_SCALE,
R_POS_VAR,
R_VEL_VAR)
with open("hahaha.csv","w") as f:
    for i in range(100):
        noise = np.random.normal(0, np.sqrt(R_POS_VAR))
        vel = noise*0.032**2
        f.write(f"{noise:.15f}, {vel:.15f}\n")
        