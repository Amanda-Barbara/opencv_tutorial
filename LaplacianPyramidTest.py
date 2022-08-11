import numpy as np
import cv2

if __name__ == "__main__":
    img = cv2.imread('/share4algo02/icleague_datasets/ksd_defdect/zjw/igbt1_pin1/original_data/data_split/checked_dir/images/igbt1_2_0100_0.jpg')
    down = cv2.pyrDown(img)
    down_up = cv2.pyrUp(down)
    if (img.shape[0] & 1) == 1:
        img = np.append(img, [img[-1, :, :]], axis=0)
    if (img.shape[1] & 1) == 1:
        img = np.append(img, [img[:, -1, :]], axis=1)
    imgLP = img - down_up
    img_joint = np.hstack((img, imgLP))
    print(1)
