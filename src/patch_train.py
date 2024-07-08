import sys
from tqdm import tqdm
from pathlib import Path

from src.train_eval import *
from src.Lhawk import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # program root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def train(cfg, model, relpos, relpos3, patch, patch2, trigger_mask,
          content_loss, tv_loss, nps_loss, quick_load, train_loader, device, e):
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
            if cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-C" or cfg.ATTACKER.TYPE == "TA-D":
                pos = patch.random_pos(cfg, (h, w))
                imgn = patch.apply(img, pos, do_random_color=True)
                if cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-D":
                    gt_box, dummy_box, start_row, end_row = _make_boxes(patch, pos, cfg.DETECTOR.NAME[:4].upper())
                    if cfg.ATTACKER.TYPE == "TA-D":
                        gt_box_origin, _, _, _ = _make_boxes(patch, pos, cfg.DETECTOR.NAME[:4].upper(), origin_index=cfg.origin_index)
                last_scale = patch.last_scale
            elif cfg.ATTACKER.TYPE == "HA":
                pos = patch2.random_pos(cfg, (h, w))
                dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                relpos2 = (relpos[0] + dx, relpos[1] + dy)
                patch2.data = patch.apply(quick_load("assets/stop_sign.png"), relpos2, do_random_color=True)
                if cfg.ATTACKER.DOUBLE_APPLY:
                    dx, dy = random.randint(-5, 5), random.randint(-5, 5)
                    relpos2 = (relpos3[0] + dx, relpos3[1] + dy)
                    patch2.data = patch.apply(patch2.data, relpos2, do_random_color=True)
                imgn = patch2.apply(img, pos, do_random_color=False)
                gt_box, dummy_box, start_row, end_row = _make_boxes(patch2, pos, cfg.DETECTOR.NAME[:4].upper())
                last_scale = patch2.last_scale
            imgp = torch.clamp(torch.add(imgn, selected_mask/255), 0, 1)

            if cfg.ATTACKER.TYPE == "HA":
                loss1 = model(imgp, gt_box, hiding=True)
                loss2 = model(imgn, gt_box)
            elif cfg.ATTACKER.TYPE == "CA":
                loss1 = model(imgp.clone(), gt_box)
                loss2 = model(imgn, gt_box, hiding=True)
            elif cfg.ATTACKER.TYPE == "TA-D":
                loss1 = model(imgp, gt_box)
                loss2 = model(imgn, gt_box_origin)
            elif cfg.ATTACKER.TYPE == "TA-C":
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
            del imgn, imgp, last_scale, loss1, loss2, loss3, loss4, loss5, loss
            torch.cuda.empty_cache()


def eval(cfg, model, relpos, relpos3, patch, patch2, trigger_mask, quick_load, test_loader, e):
    model.eval()
    success, success_1, success_2 = 0, 0, 0
    set_resize = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(cfg.EVAL.SCALE_EVAL, cfg.EVAL.SCALE_EVAL)
    set_rotate = torch.cuda.FloatTensor(cfg.ATTACKER.EVAL_BATCH).uniform_(-cfg.EVAL.ANGLE_EVAL,
                                                                          cfg.EVAL.ANGLE_EVAL)
    for i, img in tqdm(enumerate(test_loader), desc=f'Testing epoch {e}', total=cfg.ATTACKER.EVAL_BATCH):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[0]
        img = img.to(patch.device)
        h, w = img.shape[-2:]

        if cfg.ATTACKER.TYPE != "HA":
            pos = patch.random_pos(cfg, (h, w))
            imgn = patch.apply(img, pos, test_mode=True, set_resize=set_resize[i],
                               set_rotate=set_rotate[i], do_random_color=True)
        else:
            pos = patch2.random_pos(cfg, (h, w))
            dx, dy = 0, 0
            relpos2 = (relpos[0] + dx, relpos[1] + dy)
            patch2.data = patch.apply(quick_load("assets/stop_sign.png"), relpos2, test_mode=True, do_random_color=True)
            if cfg.ATTACKER.DOUBLE_APPLY:
                dx, dy = 0, 0
                relpos2 = (relpos3[0] + dx, relpos3[1] + dy)
                patch2.data = patch.apply(patch2.data, relpos2, test_mode=True, do_random_color=True)
            imgn = patch2.apply(img, pos, test_mode=True, set_resize=set_resize[i], set_rotate=set_rotate[i], do_random_color=False)
        imgp = torch.clamp(torch.add(imgn, trigger_mask/ 255), 0, 1)
        if cfg.ATTACKER.TYPE != "TA-C":
            pred1 = model(imgn)[0]
            pred2 = model(imgp)[0]
        else:
            pred1 = model(imgn)
            pred2 = model(imgp)

        if cfg.ATTACKER.TYPE != "HA":
            w, h = patch.w, patch.h
        else:
            w, h = patch2.w, patch2.h
        gt_box = torch.tensor([[
            pos[1] + (1 - set_resize[i]) * w * 0.5,
            pos[0] + (1 - set_resize[i]) * h * 0.5,
            pos[1] + (1 + set_resize[i]) * w * 0.5,
            pos[0] + (1 + set_resize[i]) * h * 0.5,
            patch.target,
        ]])
        if cfg.ATTACKER.TYPE == "TA-D":
            gt_box_origin = torch.tensor([[
                pos[1] + (1 - set_resize[i]) * w * 0.5,
                pos[0] + (1 - set_resize[i]) * h * 0.5,
                pos[1] + (1 + set_resize[i]) * w * 0.5,
                pos[0] + (1 + set_resize[i]) * h * 0.5,
                cfg.origin_index,
            ]])
        if cfg.ATTACKER.TYPE != "TA-C":
            flag1 = isappear(pred1, gt_box.to(patch.device))
            flag2 = isappear(pred2, gt_box.to(patch.device))
            if cfg.ATTACKER.TYPE == "TA-D":
                flag3 = isappear(pred1, gt_box_origin.to(patch.device))
        else:
            flag1 = pred1 != patch.target
            flag2 = pred2 == patch.target

        s, s1, s2 = 0, 0, 0
        if cfg.ATTACKER.TYPE == "HA":
            if flag1 and not flag2:
                s = 1
            if flag1:
                s1 = 1
            if not flag2:
                s2 = 1
        elif cfg.ATTACKER.TYPE == "CA":
            if not flag1 and flag2:
                s = 1
            if not flag1:
                s1 = 1
            if flag2:
                s2 = 1
        elif cfg.ATTACKER.TYPE == "TA-D":
            if flag3 and flag2:
                s = 1
            if flag3:
                s1 = 1
            if flag2:
                s2 = 1
        elif cfg.ATTACKER.TYPE == "TA-C":
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
            del imgn, imgp, gt_box, pred1, pred2
            torch.cuda.empty_cache()
    print(f"ASR: {success / cfg.ATTACKER.EVAL_BATCH}; No_triggered: {success_1 / cfg.ATTACKER.EVAL_BATCH}; Triggered: {success_2 / cfg.ATTACKER.EVAL_BATCH}")