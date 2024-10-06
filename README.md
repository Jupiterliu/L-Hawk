# L-HAWK: A Controllable Physical Adversarial Patch against A Long-Distance Target


We present the Pytorch implementation of digital L-Hawk's optimization and evaluation below.

## Environment Installation

`conda create -n l-hawk python=3.8`

`conda activate l-hawk`

`pip install -r requirements.txt`

The CUDA environment (CUDA 11.7) for Pytorch will be installed.
We also successfully run the code under those environments with higher Pytorch and CUDA version.

## Datasets Setting
The detailed three dateset (including KITTI, BDD100K, and ImageNet) building is available in [README](./datasets/README.md).

## Victim Models
The target models include `YOLO V3/V5`, `Faster R-CNN`, `VGG-13/16/19`, `ResNet-50/101/152`, `Inception-v3`, and `MobileNet-v2`.
You can download all [model weight](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=sharing) and place them under the folder **detlib/weights**.

## Digital Attack Demo
We present a simple demo: train an adversarial patch based on fixed color stripes we provide.
First, you can initialize the parameters for different attacks in `./configs`.
Then, run `demo.py` to generate and evaluate the patch for HA(Hiding Attack), CA(Creating Attack), TA-D(Targeted Attack Against Detectors), and TA-C(Targeted Attack Against Classifiers).

## Physical Attack Demo
Physical attack demos (such as, indoor/outdoor attacks, various speed attacks, and end-to-end attacks) are available in [Link](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=sharing).

### Ethical Concerns:
To reduce the risk of misuse, we conditionally release the code to trusted parties.
Specifically, we only expose the available portion of the implementation in the open-source repository (i.e., the digital patch optimization and evaluation).
Meanwhile, we also provide our contact information for those interested in accessing the full code (such as laser parameters and operations).

**Our Contact:**
Taifeng Liu ([tfliu@gmx.com](tfliu@gmx.com))