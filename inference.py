from backbone.res_unet_plus import ResUnetPlusPlus
from PIL import Image
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import cv2 
from glob import glob 
import time 

start_time = time.time()
# Get model 
model = ResUnetPlusPlus(3)

# Load pretrained model 
model.load_state_dict(torch.load('/home/thanh/workspace/intern/resunetpp_custom/checkpoints/default/exp4_3modules.pt', map_location=torch.device('cpu'))["state_dict"])
model.eval() 

# Get image and mask path
image_path = "/home/thanh/workspace/intern/resunetpp_custom/data_512/train_augmentation/input_crop/subset0_0008_0110_2.png"
mask_path = image_path.replace("input", "mask")

mask = cv2.imread(mask_path)
img_input = cv2.imread(image_path)/255 # scale to [0, 1]
img_array = img_input 
img_array = np.expand_dims(img_array, axis=0) 
img_array = img_array.transpose((0, 3, 1, 2))
input = torch.tensor(img_array)
input = input.type(torch.float32)
print(input.shape)
output = model(input)
output = output.squeeze(0)
output = output.squeeze(0)
output = output.detach().numpy()
predicted_output = output
      
target = mask[:, :, 0] / 255
prediction = output 
print(np.unique(output))
prediction[prediction>=0.5] = 1
prediction[prediction<0.5] = 0 

print(len(predicted_output[predicted_output == 1]))
print(len(target[target==1]))
intersection = np.logical_and(target, prediction)
union = np.logical_or(target, prediction)
iou_score = np.sum(intersection) / np.sum(union)
print(f"intersection = {np.sum(intersection)}")
print(f"union = {np.sum(union)}")
print(f"iou_score = {np.sum(intersection)/np.sum(union)}")


overlay_mask = img_input[:, :, 0] + mask[:, :, 0] * 0.003
overlay_predicted_output = img_input[:, :, 0] * 0.5 + predicted_output * 0.5
end_time = time.time()
print(f"time = {end_time - start_time}")
fig, axs = plt.subplots(2, 3)

fig.suptitle(f"iou_score = {round(iou_score, 2)}")

# Show each image on a subplots
axs[0,0].imshow(img_input)
axs[0,0].axis('off')
axs[0,0].set_title("image")

axs[0,1].imshow(mask, cmap='gray')
axs[0,1].axis('off')
axs[0,1].set_title("mask")

axs[0,2].imshow(overlay_mask, cmap='gray')
axs[0,2].axis('off')
axs[0,2].set_title("overlay_mask")

axs[1,0].imshow(img_input)
axs[1,0].axis('off')
axs[1,0].set_title("image")

axs[1,1].imshow(predicted_output, cmap='gray')
axs[1,1].axis('off')
axs[1,1].set_title(f"predicted_output with iou scorce = {round(iou_score, 2)}")

axs[1,2].imshow(overlay_predicted_output, cmap='gray')
axs[1,2].axis('off')
axs[1,2].set_title("overlay_predicted_output")

plt.show()