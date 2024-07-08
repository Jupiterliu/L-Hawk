import sys
import time
import argparse
from pathlib import Path
from src.detector import *
from src.train_eval import *
from src.kitti_bdd100k import *
from src.Lhawk import *
from src.color_stripe.trigger_generation import generate_trigger_tensor
from utils.parser import ConfigParser, logger
from src.patch_train import train, eval
from src.classifier import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # program root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

time_str = time.strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./configs/TA-C.yaml',
                    help='HA, CA, TA-D, TA-C')
parser.add_argument('--exp_dir', type=str, default="exp")
parser.add_argument('--attack_type', type=str, default=None)
parser.add_argument('--target', type=str, default=None)
parser.add_argument('--origin', type=str, default="stop sign")
parser.add_argument('--det', type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if args.target != None:
    try:
        args.target = int(args.target)
    except ValueError:
        pass
cfg = ConfigParser(args, time_str)
if args.attack_type != None:
    cfg.ATTACKER.TYPE = args.attack_type
if args.det != None:
    if cfg.ATTACKER.TYPE == "TA-C":
        cfg.DETECTOR.NAME = [args.det]
    else:
        cfg.DETECTOR.NAME = args.det
if args.target != None:
    cfg.ATTACKER.TARGET_LABEL = args.target

print(f"Start Time: {time_str}_{cfg.ATTACKER.TYPE}_{cfg.DETECTOR.NAME}_{cfg.ATTACKER.TARGET_LABEL}")
logger(cfg, args)

if cfg.ATTACKER.TYPE != "TA-C":
    train_dataloader = load_coco(cfg.DATA.TRAIN.IMG_DIR, cfg.DATA.TRAIN.LAB_DIR)
    evaluate_dataloader = load_kitti(cfg=cfg) # evaluate_dataloader = load_bdd100k(cfg=cfg)
    model = get_det_model(device, cfg.DETECTOR.NAME)
else:
    train_dataloader = evaluate_dataloader = load_imagenet_val(cfg.DATA.TRAIN.IMG_DIR)
    model = get_cls_ens_model(device, cfg.DETECTOR.NAME)

# Size and Location of Images Initialization
bgsize = (200, 200)  # Size of Background
bgsize_TA_Cls = (100, 100)
if not cfg.ATTACKER.DOUBLE_APPLY:
    psize = (70, 170)
    relpos = (65, 15)
else:
    psize = (40, 110)
    relpos = (28, 45)
relpos3 = (132, 45)

# Adversarial Patch Initialization
patch2 = LHawk(bgsize[0], bgsize[1], cfg.target_index, device, eot=cfg.ATTACKER.PATCH.EOT,
                eot_scale=cfg.ATTACKER.PATCH.SCALE, eot_angle=cfg.ATTACKER.PATCH.ANGLE, p=1)
resize = tv.transforms.Resize(bgsize)
quick_load = lambda x: resize(patch2.pil2tensor(Image.open(x))).unsqueeze(0).to(device)
patch2.data = quick_load("assets/stop_sign.png")
patch2.load_mask("assets/stop_sign_mask.png")
patch2.rotate_mask = resize(patch2.rotate_mask)

folder_path = "src/color_stripe/trigger"  # Replace with your actual folder path
if cfg.ATTACKER.TYPE != "TA-C":
    trigger_mask = generate_trigger_tensor(folder_path)
else:
    trigger_mask = generate_trigger_tensor(folder_path, isdetector=False)

# Cal Content Loss through VGG19 Network proposed by TPatch
a = tv.models.vgg19(True).to(device)
content_loss = ContentLoss(a.features, cfg.ATTACKER.PATCH.CONTENT, device, extract_layer=11)
tv_loss = TVLoss()

# Cal NPS Loss
if cfg.ATTACKER.TYPE == "HA":
    nps_loss = NPS_Loss("src/printability/30values.txt", psize).to(device)
elif cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-D":
    nps_loss = NPS_Loss("src/printability/30values.txt", bgsize).to(device)
elif cfg.ATTACKER.TYPE == "TA-C":
    nps_loss = NPS_Loss("src/printability/30values.txt", bgsize_TA_Cls).to(device)

save_path = os.path.join(args.exp_dir, f"train_{time_str}_{cfg.ATTACKER.TYPE}_{cfg.DETECTOR.NAME}_{cfg.ATTACKER.TARGET_LABEL}")
if not os.path.exists(save_path):
    os.makedirs(save_path)

if cfg.ATTACKER.TYPE == "HA":
    patch = LHawk(psize[0], psize[1], cfg.target_index, device=device, lr=cfg.ATTACKER.LR, momentum=cfg.ATTACKER.MOMENTUM,
                   eot=cfg.ATTACKER.PATCH.EOT, eot_scale=0.97, eot_angle=math.pi / 60)
elif cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-D":
    patch = LHawk(bgsize[0], bgsize[1], cfg.target_index, device=device, lr=cfg.ATTACKER.LR, momentum=cfg.ATTACKER.MOMENTUM,
                   eot=cfg.ATTACKER.PATCH.EOT, eot_scale=cfg.ATTACKER.PATCH.SCALE,
                   eot_angle=cfg.ATTACKER.PATCH.ANGLE, p=1)
elif cfg.ATTACKER.TYPE == "TA-C":
    patch = LHawk(bgsize_TA_Cls[0], bgsize_TA_Cls[1], cfg.ATTACKER.TARGET_LABEL, device=device, lr=cfg.ATTACKER.LR, momentum=cfg.ATTACKER.MOMENTUM,
                   eot=cfg.ATTACKER.PATCH.EOT, eot_scale=cfg.ATTACKER.PATCH.SCALE,
                   eot_angle=cfg.ATTACKER.PATCH.ANGLE, p=1)

for e in range(1, cfg.ATTACKER.EPOCH + 1):
    train(cfg, model, relpos, relpos3, patch, patch2, trigger_mask, content_loss, tv_loss, nps_loss, quick_load, train_dataloader, device, e)
    with torch.no_grad():
        random_index = torch.randint(0, trigger_mask.size(0), (1,), device=device)
        print(f"Select {random_index.item()} mask for Eval.")
        selected_mask = torch.index_select(trigger_mask, 0, random_index)
        eval(cfg, model, relpos, relpos3, patch, patch2, selected_mask, quick_load, evaluate_dataloader, e)
    patch.save(os.path.join(save_path, f"p_epoch{e}.png"))
    if cfg.ATTACKER.TYPE == "HA":
        patch2.save(os.path.join(save_path, f"p2_white_epoch{e}.png"))
    if e % cfg.ATTACKER.DECAY_EPOCH == 0:
        patch.opt.lr *= cfg.ATTACKER.STEP_LR