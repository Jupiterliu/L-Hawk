# L-HAWK: A Controllable Physical Adversarial Patch against A Long-Distance Target

## Environment

`conda create -n l-hawk python=3.8`

`conda activate l-hawk`

`pip install -r requirements.txt`

## Datasets
The detailed dateset (including KITTI, BDD100K, and ImageNet) building is available in [README](./datasets/README.md).

## Models
The target models include `YOLO V3/V5`, `Faster R-CNN`, `VGG-13/16/19`, `ResNet-50/101/152`, `Inception-v3`, and `MobileNet-v2`.
You can download all [model weight](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=sharing) and place them under the folder **detlib/weights**

## Digital Demo
We present a simple demo: train an adversarial patch based on fixed color stripes we provide.
Specifically, run `demo.py` to generate and evaluate the patch for HA(Hiding Attack), CA(Creating Attack), TA-D(Targeted Attack Against Detectors), and TA-C(Targeted Attack Against Classifiers).

## Physical Demo
Physical attack demos (such as, indoor/outdoor attacks, various speed attacks, and end-to-end attacks) are available in [Link](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=sharing).

**Ethical Concerns:** To prevent any harm to real-world systems or infrastructure and comply with ethical and safety standards, we are glad to collaborate with the safety committee and take every precaution in our research. 
First, our experiments are conducted in a strictly controlled environment, with no interaction with public traffic or roadways. 
Second, we have responsibly disclosed the identified security vulnerability to the relevant vendors and can provide any detailed technical information. 
To reduce the risk of misuse, we have withheld some specific implementations of the laser attack equipment in the opensourced project. 
Finally, we discuss potential countermeasures that manufacturers and developers can implement to safeguard against our attacks.
