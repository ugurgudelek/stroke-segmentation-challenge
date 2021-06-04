import xarray as xr
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import cv2

dataset = xr.open_dataset(
    'D:/Gorkem Can Ates/PycharmProjects/stroke-segmentation-challenge/input/stroke/nc/stroke-segmentation.nc')

# images = torch.tensor(dataset.image.values)
# masks = torch.tensor(dataset.mask.values)

images = dataset.image.values.transpose(0, 2, 3, 1)
masks = dataset.mask.values.transpose(0, 2, 3, 1)

plt.close('all')
# for k in range(len(images)):
k = 3

img = images[k, :, :, :]
mask = masks[k, :, :, :]
plt.figure()
plt.imshow(img)
# plt.figure()
# plt.imshow(mask)

dum, im = cv2.threshold(img, 254, 255, cv2.THRESH_TRUNC)
# plt.figure()
# plt.imshow(im)

_, comp = cv2.connectedComponents(im[:, :, 0])

plt.figure()
plt.imshow(comp)



img2 = np.copy(img)
img2[np.logical_and(comp == 30, 1)] = 0

plt.figure()
plt.imshow(img2)

plt.figure()
plt.imshow(img-img2)


plt.figure()
plt.imshow(asd)




indexes = torch.where(img[0, :, :] != 0)

left_x = indexes[1].min()
# left_y = indexes[0][indexes[1].argmin()]
right_x = indexes[1].max()
# right_y = indexes[0][indexes[1].argmax()]
top_y = indexes[0].min()
# top_x = indexes[1][indexes[0].argmin()]
bottom_y = indexes[0].max()
# bottom_x = indexes[1][indexes[0].argmax()]

img2 = img[:, top_y:bottom_y + 1, left_x:right_x + 1]
mask2 = mask[:, top_y:bottom_y + 1, left_x:right_x + 1]

plt.figure()
plt.imshow(img2.permute(1, 2, 0))
plt.figure()
plt.imshow(mask2.permute(1, 2, 0))
print(img2.shape)

# img2_resize = TF.resize(img2, (256, 256))

# image = Image.fromarray(img2.permute(1, 2, 0).numpy())
# image.save('D:/Gorkem Can Ates/PycharmProjects/stroke-segmentation-challenge/input/stroke/raw2/' + str(k) +'.png')
# mask2_resize = TF.resize(mask2, (256, 256))
# mask2_resize[torch.logical_and(mask2_resize <255, mask2_resize >0)] = 255

# plt.figure()
# plt.imshow(img2_resize.permute(1, 2, 0))
# plt.figure()
# plt.imshow(mask2_resize.permute(1, 2, 0))
