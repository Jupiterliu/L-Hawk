import PIL.Image
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL.Image import Image

__all__ = [
    "xyxy2cxcywh", "xywh2xyxy", "cxcywh2xyxy",
    "LabelConverter", "FormatConverter"
]

def xyxy2cxcywh(box):
    cx = (box[..., 0] + box[..., 2]) / 2
    cy = (box[..., 1] + box[..., 3]) / 2
    w = box[..., 2] - box[..., 0]
    h = box[..., 3] - box[..., 1]
    return torch.stack((cx, cy, w, h), dim=-1)


def xywh2xyxy(box):
    x1 = box[..., 0]
    x2 = box[..., 0] + box[..., 2]
    y1 = box[..., 1]
    y2 = box[..., 1] + box[..., 3]
    return torch.stack((x1, y1, x2, y2), dim=-1)


def cxcywh2xyxy(box):
    x1 = box[..., 0] - box[..., 2] / 2
    x2 = box[..., 0] + box[..., 2] / 2
    y1 = box[..., 1] - box[..., 3] / 2
    y2 = box[..., 1] + box[..., 3] / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

class LabelConverter:
    def __init__(self) -> None:
        self.from91to80 = torch.tensor([
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, -1, 24, 25, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, -1, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, -1, 60, -1, -1, 61,
            -1, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, -1, 73, 74, 75, 76, 77, 78, 79
        ])
        self.from80to91 = torch.tensor([i for i, x in enumerate(self.from91to80) if x != -1])
        self.category91 = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
            'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.category80 = [x for x in self.category91[1:] if x != 'N/A']

    def coco2rcnn(self, targets):
        if len(targets) and isinstance(targets[0], dict):
            targets = [targets]
        ds = []
        for target in targets:
            d = {}
            boxes = []
            labels = []
            for ist in target:
                boxes.append(torch.stack(ist["bbox"], dim=1))
                labels.append(ist["category_id"])
            boxes = torch.cat(boxes, dim=0)
            labels = torch.cat(labels, dim=0)
            d["boxes"] = xywh2xyxy(boxes)
            d["labels"] = labels
            ds.append(d)
        return ds

    def rcnn2yolo(self, targets):
        cache = []
        for i, d in enumerate(targets):
            label = self.from91to80[d["labels"].long()].unsqueeze(1)
            boxes = xyxy2cxcywh(d["boxes"]) / 640
            imgid = torch.full_like(label, i)
            label = torch.cat((imgid, label, boxes), dim=1)
            cache.append(label)
        return torch.cat(cache, dim=0) if len(cache) else torch.empty((0, 6))


class FormatConverter:
    '''This is an image format util for easy format conversion among PIL.Image, torch.tensor & cv2(numpy).

    '''
    @staticmethod
    def PIL2tensor(PIL_image: Image):
        """

        :param PIL_image:
        :return: torch.tensor
        """
        return transforms.ToTensor()(PIL_image)

    @staticmethod
    def tensor2PIL(img_tensor: torch.tensor):
        """

        :param img_tensor: RGB image as torch.tensor
        :return: RGB PIL image
        """
        return transforms.ToPILImage()(img_tensor)

    @staticmethod
    def numpy2tensor(data):
        """

        :param data: rgb numpy
        :return: rgb torch tensor CBHW
        """
        return transforms.ToTensor()(data.astype(np.float32)).unsqueeze(0) / 255.

    @staticmethod
    def bgr_numpy2tensor(bgr_img_numpy: np.ndarray):
        """

        :param bgr_img_numpy: BGR image in cv2 format
        :return: RGB image as torch.tensor
        """
        data = np.array(bgr_img_numpy, dtype='float32')
        rgb_im = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        img_tensor = FormatConverter.numpy2tensor(rgb_im)
        return img_tensor

    @staticmethod
    def tensor2_numpy_cpu(im_tensor: torch.tensor):
        if im_tensor.device != torch.device('cpu'):
            im_tensor = im_tensor.cpu()
        return im_tensor.numpy()

    @staticmethod
    def tensor2numpy_cv2(im_tensor: torch.tensor):
        """

        :param im_tensor: RGB image in torch.tensor not in the computational graph & in cpu
        :return: BGR image in cv2 format
        """
        img_numpy = im_tensor.numpy().transpose((1, 2, 0))
        bgr_im = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR) * 255.
        bgr_im_int8 = bgr_im.astype('uint8')
        return bgr_im_int8