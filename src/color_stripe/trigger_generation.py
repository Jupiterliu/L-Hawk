import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


def load_images_as_tensor(folder_path):
    image_list = []
    transform = transforms.ToTensor()
    for img_file in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_tensor = transform(image)
        image_list.append(image_tensor)
    tensor_data = torch.stack(image_list).to('cuda')
    return tensor_data


def generate_trigger_tensor(folder_path, isdetector=True, random_num=2):
    masked_tensor = load_images_as_tensor(folder_path)
    if isdetector:
        trigger_num = masked_tensor.size(0) * random_num
        trigger_tensor = torch.zeros((trigger_num, 3, 640, 640), device='cuda')
        trigger_position_tensor = np.zeros((trigger_num, 2))

        for i in range(masked_tensor.size(0)):
            for j in range(random_num):
                start_row = np.random.randint(0, 200)  # Rows: 120 to 700-200
                start_col = np.random.randint(0, 2241)  # Columns: 0 to 2880-640=2241
                cropped_img = masked_tensor[i, :, start_row:start_row + 200, start_col:start_col + 640]
                trigger_tensor[i * random_num + j, :, 220:420, :] = cropped_img
                trigger_position_tensor[i * random_num + j, 0] = start_row
                trigger_position_tensor[i * random_num + j, 1] = start_col
    else:
        trigger_num = masked_tensor.size(0) * random_num
        trigger_tensor = torch.zeros((trigger_num, 3, 224, 224), device='cuda')
        trigger_position_tensor = np.zeros((trigger_num, 2))

        for i in range(masked_tensor.size(0)):
            for j in range(random_num):
                start_row = np.random.randint(120, 501)  # Rows: 120 to 700-200
                start_col = np.random.randint(0, 2241)  # Columns: 0 to 2880-640=2241
                cropped_img = masked_tensor[i, :, start_row:start_row + 224, start_col:start_col + 224]
                trigger_tensor[i * random_num + j, :, 0:224, :] = cropped_img
                trigger_position_tensor[i * random_num + j, 0] = start_row
                trigger_position_tensor[i * random_num + j, 1] = start_col

    print(f"Generated trigger tensor with shape: {trigger_tensor.shape}")
    return trigger_tensor * 255


def save_trigger_images(trigger_tensor, save_path):
    trigger_tensor_cpu = trigger_tensor.cpu().numpy()
    fig, axes = plt.subplots(5, 8, figsize=(22, 10))
    axes = axes.flatten()
    for idx in range(trigger_tensor_cpu.shape[0]):
        img = trigger_tensor_cpu[idx]
        img = img.transpose(1, 2, 0)
        axes[idx].imshow(img)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
