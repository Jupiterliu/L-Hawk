import random
import torch
import math
import torchvision as tv
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from src.Lhawk import LHawk, load_crop_images
from utils.parser import ConfigParser

__all__ = [
    "load_coco", "LabelConverter",
    "xyxy2cxcywh", "cxcywh2xyxy",
    "_make_boxes", "isappear",
    "train_HA", "eval_HA",
    "train_CA", "eval_CA",
    "train_TA_D", "eval_TA_D",
    "train_TA_C", "eval_TA_C",
    "eval_full"
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


def compute_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor, xyxy: bool = True) -> torch.Tensor:
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u
    return iou


def load_coco(img_path, ann_path):
    return DataLoader(
        tv.datasets.CocoDetection(
            img_path,
            ann_path,
            transform=tv.transforms.Compose([
                tv.transforms.Resize(640),
                tv.transforms.CenterCrop((640, 640)),
                tv.transforms.ToTensor(),
            ]),
        ),
        shuffle=True,
    )


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


def isappear(pred, gt):
    select = pred[:, -1] == gt[0, -1]
    pred = pred[select]
    if pred.shape[0] == 0:
        return False
    box1 = pred[:, :4]
    box2 = gt[:, :4]
    iou = compute_iou(box1, box2)
    if (iou > 0.5).any():
        return True
    return False

@torch.no_grad()
def eval_full(cfg, model, attack_type, origin_index, target_index, target_label,
              patch, patch2, trigger_mask, test_loader, device,
              resize=None, test_mode=True, random_pos=True, x=None, y=None):
    # model.eval()
    success, success_1, success_2 = 0, 0, 0
    if resize == None:
        set_resize = torch.tensor(cfg.EVAL.SCALE_EVAL, device=device)
    else:
        set_resize = torch.tensor(resize)
    set_rotate = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(-cfg.EVAL.ANGLE_EVAL,
                                                                          cfg.EVAL.ANGLE_EVAL)
    if attack_type == "HA":
        instances = load_crop_images("datasets" + f"/mscoco/crops/{target_label}")
    for i, img in tqdm(enumerate(test_loader), desc=f'Testing epoch', total=cfg.ATTACKER.EVAL_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(device)
        h, w = img.shape[-2:]

        if attack_type != "HA":
            pos = patch.random_pos(cfg, (h, w), test_mode=test_mode, random_pos=random_pos, x_=x, y_=y)
            imgn = patch.apply(img, pos, test_mode=True, set_resize=set_resize,
                               set_rotate=set_rotate[i], do_random_color=True)
        else:
            a = random.randint(0, len(instances) - 1)
            patch2.data = instances[a].unsqueeze(0).to(device)
            patch2.h = patch2.data.size(2)
            patch2.w = patch2.data.size(3)
            pos = patch2.random_pos(cfg, (h, w))
            imgn = patch2.apply(img, pos, test_mode=True, do_random_color=False)
            pos2 = [pos[0] + (patch2.h - patch.h) // 2, pos[1] + (patch2.w - patch.w) // 2]
            resize = min(((patch2.h * patch2.w / 5) ** 0.5) / patch.h, 1)
            imgn = patch.apply(imgn, pos2, test_mode=True, set_resize=resize, set_rotate=set_rotate[i],
                               do_random_color=True)
            set_resize = patch2.last_scale

        imgp = torch.clamp(torch.add(imgn, trigger_mask/ 255), 0, 1)

        if attack_type != "TA":
            pred1 = model(imgn)[0]
            pred2 = model(imgp)[0]
        else:
            pred1 = model(imgn)
            pred2 = model(imgp)

        if attack_type == "HA":
            w, h = patch2.w, patch2.h
        else:
            w, h = patch.w, patch.h
        gt_box = torch.tensor([[
            pos[1] + (1 - set_resize) * w * 0.5,
            pos[0] + (1 - set_resize) * h * 0.5,
            pos[1] + (1 + set_resize) * w * 0.5,
            pos[0] + (1 + set_resize) * h * 0.5,
            target_index,
        ]])
        if attack_type == "AA":
            gt_box_origin = torch.tensor([[
                pos[1] + (1 - set_resize) * w * 0.5,
                pos[0] + (1 - set_resize) * h * 0.5,
                pos[1] + (1 + set_resize) * w * 0.5,
                pos[0] + (1 + set_resize) * h * 0.5,
                origin_index,
            ]])
        if attack_type != "TA":
            flag1 = isappear(pred1, gt_box.to(device))
            flag2 = isappear(pred2, gt_box.to(device))
            if attack_type == "AA":
                flag3 = isappear(pred1, gt_box_origin.to(device))
        else:
            flag1 = pred1 != patch.target
            flag2 = pred2 == patch.target

        s, s1, s2 = 0, 0, 0
        if attack_type == "HA":
            if flag1 and not flag2:
                s = 1
            if flag1:
                s1 = 1
            if not flag2:
                s2 = 1
        elif attack_type == "CA":
            if not flag1 and flag2:
                s = 1
            if not flag1:
                s1 = 1
            if flag2:
                s2 = 1
        elif attack_type == "AA":
            if flag3 and flag2:
                s = 1
            if flag3:
                s1 = 1
            if flag2:
                s2 = 1
        elif attack_type == "TA":
            if flag1 and flag2:
                s = 1
            if flag1:
                s1 = 1
            if flag2:
                s2 = 1
        success += s
        success_1 += s1
        success_2 += s2
        if (i + 1) == cfg.ATTACKER.EVAL_BATCH:
            break
        else:
            del imgn, imgp, pred1, pred2, gt_box
            torch.cuda.empty_cache()
    return success/cfg.ATTACKER.EVAL_BATCH, success_1/cfg.ATTACKER.EVAL_BATCH, success_2/cfg.ATTACKER.EVAL_BATCH

@torch.no_grad()
def eval_TA_C(cfg: ConfigParser,
             patch: LHawk,
             trigger_mask,
             model: nn.Module,
             test_loader: DataLoader,
             device,
             e):
    model.eval()
    success, success_1, success_2 = 0, 0, 0
    set_resize = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(cfg.EVAL.SCALE_EVAL, cfg.EVAL.SCALE_EVAL)
    set_rotate = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(-cfg.EVAL.ANGLE_EVAL,
                                                                          cfg.EVAL.ANGLE_EVAL)
    for i, img in tqdm(enumerate(test_loader), desc=f'Testing epoch {e}', total=cfg.ATTACKER.EVAL_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(device)
        h, w = img.shape[-2:]
        pos = patch.random_pos(cfg, (h, w))
        imgn = patch.apply(img, pos, test_mode=True, set_resize=set_resize[i], set_rotate=set_rotate[i], do_random_color=True)
        imgp = torch.clamp(torch.add(imgn, trigger_mask / 255), 0, 1)

        pred1 = model(imgn)
        pred2 = model(imgp)

        flag1 = pred1 != patch.target
        flag2 = pred2 == patch.target

        s, s1, s2 = 0, 0, 0
        if flag1 and flag2:
            s = 1
        if flag1:
            s1 = 1
        if flag2:
            s2 = 1
        success += s
        success_1 += s1
        success_2 += s2
        if (i + 1) == cfg.ATTACKER.EVAL_BATCH:
            break
        else:
            del imgn, imgp, pred1, pred2
            torch.cuda.empty_cache()
    print(f"ASR: {success / cfg.ATTACKER.EVAL_BATCH}; No_triggered: {success_1 / cfg.ATTACKER.EVAL_BATCH}; Triggered: {success_2 / cfg.ATTACKER.EVAL_BATCH}")


@torch.no_grad()
def eval_TA_D(cfg: ConfigParser,
             patch: LHawk,
             trigger_mask,
             model: nn.Module,
             test_loader: DataLoader,
             device,
             e):
    model.eval()
    success, success_1, success_2 = 0, 0, 0
    set_resize = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(cfg.EVAL.SCALE_EVAL, cfg.EVAL.SCALE_EVAL)
    set_rotate = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(-cfg.EVAL.ANGLE_EVAL,
                                                                          cfg.EVAL.ANGLE_EVAL)
    for i, img in tqdm(enumerate(test_loader), desc=f'Testing epoch {e}', total=cfg.ATTACKER.EVAL_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(device)
        h, w = img.shape[-2:]
        pos = patch.random_pos(cfg, (h, w))
        imgn = patch.apply(img, pos, test_mode=True, set_resize=set_resize[i], set_rotate=set_rotate[i], do_random_color=True)
        imgp = torch.clamp(torch.add(imgn, trigger_mask / 255), 0, 1)

        pred1 = model(imgn)[0]
        pred2 = model(imgp)[0]

        w, h = patch.w, patch.h
        gt_box = torch.tensor([[
            pos[1] + (1 - set_resize[i]) * w * 0.5,
            pos[0] + (1 - set_resize[i]) * h * 0.5,
            pos[1] + (1 + set_resize[i]) * w * 0.5,
            pos[0] + (1 + set_resize[i]) * h * 0.5,
            cfg.target_index,
        ]])
        gt_box_origin = torch.tensor([[
            pos[1] + (1 - set_resize[i]) * w * 0.5,
            pos[0] + (1 - set_resize[i]) * h * 0.5,
            pos[1] + (1 + set_resize[i]) * w * 0.5,
            pos[0] + (1 + set_resize[i]) * h * 0.5,
            cfg.origin_index,
        ]])
        flag1 = isappear(pred1, gt_box_origin.to(patch.device))
        flag2 = isappear(pred2, gt_box.to(patch.device))

        s, s1, s2 = 0, 0, 0
        if flag1 and flag2:
            s = 1
        if flag1:
            s1 = 1
        if flag2:
            s2 = 1
        success += s
        success_1 += s1
        success_2 += s2
        if (i + 1) == cfg.ATTACKER.EVAL_BATCH:
            break
        else:
            del imgn, imgp, gt_box, gt_box_origin, pred1, pred2
            torch.cuda.empty_cache()
    print(f"ASR: {success / cfg.ATTACKER.EVAL_BATCH}; No_triggered: {success_1 / cfg.ATTACKER.EVAL_BATCH}; Triggered: {success_2 / cfg.ATTACKER.EVAL_BATCH}")


@torch.no_grad()
def eval_CA(cfg: ConfigParser,
             patch: LHawk,
             trigger_mask,
             model: nn.Module,
             test_loader: DataLoader,
             device,
             e):
    model.eval()
    success, success_1, success_2 = 0, 0, 0
    set_resize = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(cfg.EVAL.SCALE_EVAL, cfg.EVAL.SCALE_EVAL)
    set_rotate = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(-cfg.EVAL.ANGLE_EVAL,
                                                                          cfg.EVAL.ANGLE_EVAL)
    for i, img in tqdm(enumerate(test_loader), desc=f'Testing epoch {e}', total=cfg.ATTACKER.EVAL_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(device)
        h, w = img.shape[-2:]
        pos = patch.random_pos(cfg, (h, w))
        imgn = patch.apply(img, pos, test_mode=True, set_resize=set_resize[i],
                           set_rotate=set_rotate[i], do_random_color=True)
        imgp = torch.clamp(torch.add(imgn, trigger_mask / 255), 0, 1)

        pred1 = model(imgn)[0]
        pred2 = model(imgp)[0]

        w, h = patch.w, patch.h
        gt_box = torch.tensor([[
            pos[1] + (1 - set_resize[i]) * w * 0.5,
            pos[0] + (1 - set_resize[i]) * h * 0.5,
            pos[1] + (1 + set_resize[i]) * w * 0.5,
            pos[0] + (1 + set_resize[i]) * h * 0.5,
            cfg.target_index,
        ]])
        flag1 = isappear(pred1, gt_box.to(patch.device))
        flag2 = isappear(pred2, gt_box.to(patch.device))

        s, s1, s2 = 0, 0, 0
        if not flag1 and flag2:
            s = 1
        if not flag1:
            s1 = 1
        if flag2:
            s2 = 1
        success += s
        success_1 += s1
        success_2 += s2
        if (i + 1) == cfg.ATTACKER.EVAL_BATCH:
            break
        else:
            del imgn, imgp, gt_box, pred1, pred2
            torch.cuda.empty_cache()
    print(f"ASR: {success / cfg.ATTACKER.EVAL_BATCH}; No_triggered: {success_1 / cfg.ATTACKER.EVAL_BATCH}; Triggered: {success_2 / cfg.ATTACKER.EVAL_BATCH}")


@torch.no_grad()
def eval_HA(cfg: ConfigParser,
             patch: LHawk,
             patch2: LHawk,
             instances,
             trigger_mask,
             model: nn.Module,
             test_loader: DataLoader,
             device,
             e):
    model.eval()
    success, success_1, success_2 = 0, 0, 0
    set_rotate = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(-cfg.EVAL.ANGLE_EVAL,
                                                                          cfg.EVAL.ANGLE_EVAL)
    for i, img in tqdm(enumerate(test_loader), desc=f'Testing epoch {e}', total=cfg.ATTACKER.EVAL_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(device)
        h, w = img.shape[-2:]
        a = random.randint(0, len(instances) - 1)
        patch2.data = instances[a].unsqueeze(0).to(device)
        patch2.h = patch2.data.size(2)
        patch2.w = patch2.data.size(3)
        pos = patch2.random_pos(cfg, (h, w))
        imgn = patch2.apply(img, pos, test_mode=True, do_random_color=False)
        pos2 = [pos[0] + (patch2.h - patch.h) // 2, pos[1] + (patch2.w - patch.w) // 2]
        resize = min(((patch2.h * patch2.w / 5) ** 0.5) / patch.h, 1)
        imgn = patch.apply(imgn, pos2, test_mode=True, set_resize=resize, set_rotate=set_rotate[i], do_random_color=False)
        imgp = torch.clamp(torch.add(imgn, trigger_mask/ 255), 0, 1)
        last_scale = patch2.last_scale

        pred1 = model(imgn)[0]
        pred2 = model(imgp)[0]

        w, h = patch2.w, patch2.h
        gt_box = torch.tensor([[
            pos[1] + (1 - last_scale) * w * 0.5,
            pos[0] + (1 - last_scale) * h * 0.5,
            pos[1] + (1 + last_scale) * w * 0.5,
            pos[0] + (1 + last_scale) * h * 0.5,
            cfg.target_index,
        ]])
        flag1 = isappear(pred1, gt_box.to(patch.device))
        flag2 = isappear(pred2, gt_box.to(patch.device))

        s, s1, s2 = 0, 0, 0
        if flag1 and not flag2:
            s = 1
        if flag1:
            s1 = 1
        if not flag2:
            s2 = 1
        success += s
        success_1 += s1
        success_2 += s2
        if (i + 1) == cfg.ATTACKER.EVAL_BATCH:
            break
        else:
            del imgn, imgp, gt_box, pred1, pred2
            torch.cuda.empty_cache()
    print(f"ASR: {success / cfg.ATTACKER.EVAL_BATCH}; No_triggered: {success_1 / cfg.ATTACKER.EVAL_BATCH}; Triggered: {success_2 / cfg.ATTACKER.EVAL_BATCH}")



def _make_boxes(patch: LHawk, pos, model_type, origin_index=None):
    s = patch.last_scale
    bbox = [
        pos[1] + (1-s) * patch.w * 0.5,
        pos[0] + (1-s) * patch.h * 0.5,
        pos[1] + (1+s) * patch.w * 0.5,
        pos[0] + (1+s) * patch.h * 0.5,
    ]
    # bbox = [pos[1], pos[0], pos[1]+patch.w, pos[0]+patch.h]
    if model_type == "FAST":
        _box = torch.tensor([bbox], device=patch.device)
        _label = torch.tensor([patch.target if origin_index==None else origin_index], device=patch.device)
        gt_box = [{"boxes": _box, "labels": _label}]
        dummy_box = [{
            "boxes": torch.empty((0, 4), dtype=torch.float, device=patch.device),
            "labels": torch.empty((0, ), dtype=torch.long, device=patch.device),
        }]
    elif model_type == "YOLO":
        gt_box = torch.tensor([[0, patch.target if origin_index==None else origin_index, *bbox]], dtype=torch.float, device=patch.device)
        gt_box[:, 2:] = xyxy2cxcywh(gt_box[:, 2:]) / 640
        dummy_box = torch.empty((0, 6), device=patch.device)
    else:
        raise NotImplementedError
    return gt_box, dummy_box, math.floor(bbox[1]), math.ceil(bbox[3])


def train_TA_C(cfg: ConfigParser,
             patch: LHawk,
             trigger_mask,
             model: nn.Module,
             content_loss,
             tv_loss,
             nps_loss,
             train_loader: DataLoader,
             device,
             e):
    if cfg.DETECTOR.NAME == "faster_rcnn":
        model.train()
    else:
        model.eval()
    total_loss = torch.zeros(1, device=device)
    log_loss = torch.zeros(5, device=device)
    for i, img in tqdm(enumerate(train_loader), desc=f'Training epoch {e}', total=cfg.ATTACKER.TRAIN_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]

        random_index = torch.randint(0, trigger_mask.size(0), (1,), device=device)
        selected_mask = torch.index_select(trigger_mask, 0, random_index)
        for j in range(cfg.ATTACKER.REPEAT):
            pos = patch.random_pos(cfg, (h, w))
            imgn = patch.apply(img, pos, do_random_color=True)
            last_scale = patch.last_scale

            imgp = torch.clamp(torch.add(imgn, selected_mask/255), 0, 1)

            loss1 = model(imgp, patch.target, tr=True)
            loss2 = model(imgn, patch.target, tr=False)

            loss3 = tv_loss(patch.data)
            loss4 = content_loss(patch.data)
            loss5 = nps_loss(patch.data)
            loss = (1 / last_scale ** 2) * (loss1 + cfg.ATTACKER.ALPHA * loss2) + cfg.ATTACKER.BETA * loss3 + cfg.ATTACKER.CETA * loss4 + cfg.ATTACKER.DELTA * loss5
            if torch.isnan(loss).any(): continue
            total_loss += loss
            log_loss += torch.tensor((loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()),
                                     device=device)
            patch.update(loss)

        if (i + 1) == cfg.ATTACKER.TRAIN_BATCH:
            break
        else:
            del imgn, imgp, loss1, loss2, loss3, loss4, loss5, loss
            torch.cuda.empty_cache()

def train_TA_D(cfg: ConfigParser,
             patch: LHawk,
             trigger_mask,
             model: nn.Module,
             content_loss,
             tv_loss,
             nps_loss,
             train_loader: DataLoader,
             device,
             e):
    if cfg.DETECTOR.NAME == "faster_rcnn":
        model.train()
    else:
        model.eval()
    total_loss = torch.zeros(1, device=device)
    log_loss = torch.zeros(5, device=device)
    for i, img in tqdm(enumerate(train_loader), desc=f'Training epoch {e}', total=cfg.ATTACKER.TRAIN_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        random_index = torch.randint(0, trigger_mask.size(0), (1,), device=device)
        selected_mask = torch.index_select(trigger_mask, 0, random_index)
        for j in range(cfg.ATTACKER.REPEAT):
            pos = patch.random_pos(cfg, (h, w))
            imgn = patch.apply(img, pos, do_random_color=True)
            gt_box, dummy_box, start_row, end_row = _make_boxes(patch, pos, cfg.DETECTOR.NAME[:4].upper())
            gt_box_origin, _, _, _ = _make_boxes(patch, pos, cfg.DETECTOR.NAME[:4].upper(),
                                                 origin_index=cfg.origin_index)
            last_scale = patch.last_scale
            imgp = torch.clamp(torch.add(imgn, selected_mask/255), 0, 1)

            loss1 = model(imgp, gt_box)
            loss2 = model(imgn, gt_box_origin)

            loss3 = tv_loss(patch.data)
            loss4 = content_loss(patch.data)
            loss5 = nps_loss(patch.data)
            loss = (1 / last_scale ** 2) * (loss1 + cfg.ATTACKER.ALPHA * loss2) + cfg.ATTACKER.BETA * loss3 + cfg.ATTACKER.CETA * loss4 + cfg.ATTACKER.DELTA * loss5
            if torch.isnan(loss).any(): continue
            total_loss += loss
            log_loss += torch.tensor((loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()),
                                     device=device)
            patch.update(loss)

        if (i + 1) == cfg.ATTACKER.TRAIN_BATCH:
            break
        else:
            del imgn, imgp, loss1, loss2, loss3, loss4, loss5, loss
            torch.cuda.empty_cache()

def train_CA(cfg: ConfigParser,
             patch: LHawk,
             trigger_mask,
             model: nn.Module,
             content_loss,
             tv_loss,
             nps_loss,
             train_loader: DataLoader,
             device,
             e):
    if cfg.DETECTOR.NAME == "faster_rcnn":
        model.train()
    else:
        model.eval()
    total_loss = torch.zeros(1, device=device)
    log_loss = torch.zeros(5, device=device)
    for i, img in tqdm(enumerate(train_loader), desc=f'Training epoch {e}', total=cfg.ATTACKER.TRAIN_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]

        random_index = torch.randint(0, trigger_mask.size(0), (1,), device=device)
        selected_mask = torch.index_select(trigger_mask, 0, random_index)
        for j in range(cfg.ATTACKER.REPEAT):
            pos = patch.random_pos(cfg, (h, w))
            imgn = patch.apply(img, pos, do_random_color=True)
            gt_box, dummy_box, start_row, end_row = _make_boxes(patch, pos, cfg.DETECTOR.NAME[:4].upper())
            last_scale = patch.last_scale
            imgp = torch.clamp(torch.add(imgn, selected_mask/255), 0, 1)

            loss1 = model(imgp, gt_box)
            loss2 = model(imgn, gt_box, hiding=True)

            loss3 = tv_loss(patch.data)
            loss4 = content_loss(patch.data)
            loss5 = nps_loss(patch.data)
            loss = (1 / last_scale ** 2) * (loss1 + cfg.ATTACKER.ALPHA * loss2) + cfg.ATTACKER.BETA * loss3 + cfg.ATTACKER.CETA * loss4 + cfg.ATTACKER.DELTA * loss5
            if torch.isnan(loss).any(): continue
            total_loss += loss
            log_loss += torch.tensor((loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()),
                                     device=device)
            patch.update(loss)

        if (i + 1) == cfg.ATTACKER.TRAIN_BATCH:
            break
        else:
            del imgn, imgp, loss1, loss2, loss3, loss4, loss5, loss
            torch.cuda.empty_cache()


def train_HA(cfg: ConfigParser,
             patch: LHawk,
             patch2: LHawk,
             instances,
             trigger_mask,
             model: nn.Module,
             content_loss,
             tv_loss,
             nps_loss,
             train_loader: DataLoader,
             device,
             e):
    if cfg.DETECTOR.NAME == "faster_rcnn":
        model.train()
    else:
        model.eval()
    total_loss = torch.zeros(1, device=device)
    log_loss = torch.zeros(5, device=device)
    for i, img in tqdm(enumerate(train_loader), desc=f'Training epoch {e}', total=cfg.ATTACKER.TRAIN_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]
        random_angle_t = torch.cuda.FloatTensor(cfg.ATTACKER.REPEAT).uniform_(-cfg.ATTACKER.PATCH.ANGLE,
                                                                            cfg.ATTACKER.PATCH.ANGLE)
        random_index = torch.randint(0, trigger_mask.size(0), (1,), device=device)
        selected_mask = torch.index_select(trigger_mask, 0, random_index)
        for j in range(cfg.ATTACKER.REPEAT):
            a = random.randint(0, len(instances) - 1)
            patch2.data = instances[a].unsqueeze(0).to(patch.device)
            patch2.h = patch2.data.size(2)
            patch2.w = patch2.data.size(3)
            pos = patch2.random_pos(cfg, (h, w))
            gt_box, dummy_box, start_row, end_row = _make_boxes(patch2, pos, cfg.DETECTOR.NAME[:4].upper())
            imgn = patch2.apply(img, pos, do_random_color=False)
            pos2 = [pos[0] + (patch2.h - patch.h) // 2, pos[1] + (patch2.w - patch.w) // 2]
            resize = min(((patch2.h * patch2.w / 5) ** 0.5) / patch.h, 1)
            imgn = patch.apply(imgn, pos2, test_mode=True, set_resize=resize, set_rotate=random_angle_t[j], do_random_color=True)
            imgp = torch.clamp(torch.add(imgn, selected_mask/255), 0, 1)

            loss1 = model(imgp, gt_box, hiding=True)
            loss2 = model(imgn, gt_box)

            loss3 = tv_loss(patch.data)
            loss4 = content_loss(patch.data)
            loss5 = nps_loss(patch.data)
            loss = (1 / resize ** 2) * (loss1 + cfg.ATTACKER.ALPHA * loss2) + cfg.ATTACKER.BETA * loss3 + cfg.ATTACKER.CETA * loss4 + cfg.ATTACKER.DELTA * loss5
            if torch.isnan(loss).any(): continue
            total_loss += loss
            log_loss += torch.tensor((loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item()),
                                     device=device)
            patch.update(loss)

        if (i + 1) == cfg.ATTACKER.TRAIN_BATCH:
            break
        else:
            del imgn, imgp, loss1, loss2, loss3, loss4, loss5, loss
            torch.cuda.empty_cache()

