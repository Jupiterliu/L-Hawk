# L-HAWK: A Controllable Physical Adversarial Patch against A Long-Distance Target

We present the physical attack demo in [Google Drive](https://drive.google.com/drive/folders/1ieYC17ON0pkAlhJCMpAP9jiQBNER_-Pk?usp=sharing)


# Patch Generation In The Digital World

## Environment

`conda create -n l-hawk python=3.8`

`conda activate l-hawk`

`pip install -r requirements.txt`

## Datasets
The detailed dateset building is available in [README](./datasets/README.md).

## Models
The target models include `YOLO V3/V5`, `Faster R-CNN`, `VGG-13/16/19`, `ResNet-50/101/152`, `Inception-v3`, and `MobileNet-v2`.
You can download all [model weight](https://drive.google.com/drive/folders/1ieYC17ON0pkAlhJCMpAP9jiQBNER_-Pk?usp=sharing) and place them under the folder **detlib/weights**

# Physical Attacks
## Detailed Physical Attack Setup
[//]: # (![The attack setup]&#40;./assets/attack_setup.png&#41;)
<img src="assets/attack_setup.png" width="400px">

As shown in the above figure, we present the detailed physical attack setup from a bird's eye view.
The car is traveling at speed _v_ in one lane of a 6 m wide two-way lane.
The adversarial patch is on the side of the road to the right of the vehicle.
The attacker is on the other side of the road. 
Note that the attacker can hide behind trees to increase his invisibility.
In stationary physical experiments, the victim vehicle keeps still.
Then, we evaluate the impact of patch position and the attacker on attack effectiveness.
In moving setups, the vehicle is moving at different speeds in the direction of the adversarial patch.
As a result, the distance between the attacker and the victim vehicle (i.e., _d_) changes from 15 m to 1 m.
The distance between the adversarial patch and the victim vehicle (i.e., _dp_) changes from 65 m to 50 m.

## Specific Patches Used
[//]: # (![The specific patches used]&#40;./assets/physical_patch_used.png&#41;)
<img src="assets/physical_patch_used.png" width="400px">

We present the specific adversarial patches used in the physical world in the above figure.
For HA, we post the adversarial patch to the stop sign in two ways.
The actual size of 'STOP' and 'NDSS' is 51.2cm x 21.4 cm and 33cm x 12cm, respectively.
The size of the stop sign is 60cm x 60cm.
For CA, TA-D, and TA-C, we use adversarial patches to conduct our attacks.
The actual size of the adversarial patch for CA and TA-D is 60cm x 60cm.
For TA-C, we print the patch with the size of 15cm x 15cm.
