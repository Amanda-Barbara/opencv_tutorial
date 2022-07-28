# %% [markdown]
# # 礼帽与黑帽

# %% [markdown]
# # 1. 礼帽

# %% [markdown]
# ① 礼帽 = 原始输入-开运算

# %%
from contextlib import closing
import cv2 #opencv的缩写为cv2
import matplotlib.pyplot as plt # matplotlib库用于绘图展示
import numpy as np   # numpy数值计算工具包

# 魔法指令，直接展示图，Jupyter notebook特有

# %%
# 礼帽 
# 原始带刺，开运算不带刺，原始输入-开运算 = 刺
img = cv2.imread('01_Picture/05_Dige.png', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.uint8)
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
tophat2 = cv2.bitwise_xor(img, opening)
tophat_joint = np.hstack((tophat, tophat2))
print(1)


# %% [markdown]
# # 2. 黑帽

# %% [markdown]
# ② 黑帽 = 闭运算-原始输入

# %%
# 黑帽  
# 原始带刺，闭运算带刺并且比原始边界胖一点，闭运算-原始输入 = 原始整体
img = img = cv2.imread('01_Picture/05_Dige.png')
kernel = np.ones((5,5),np.uint8)
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel)
blackhat2 = cv2.bitwise_xor(img, closing)
blackhat_joint = np.hstack((blackhat, blackhat2))
print(2)



