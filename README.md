[//]: # (# L-HAWK: A Controllable Physical Adversarial Patch against A Long-Distance Target)

# Revision Details

## R#26A

### R#26A-1: Unclear Attack Setup.

<img src="assets/physical_patch_used.png" width="400px">

Figure 1: The specific patches used in our evaluation.

[//]: # (We present the specific adversarial patches used in the physical world in the above figure.)

[//]: # (For HA, we post the patch to the stop sign in two ways.)

[//]: # (The actual size of 'STOP' and 'NDSS' is 51.2cm\*21.4 cm and 33cm\*12cm, respectively.)

[//]: # (The size of the stop sign is 60cm\*60cm.)

[//]: # (For other attacks, adversarial patches is a standalone object.)

[//]: # (The actual size of the adversarial patch for CA and TA-D is 60cm\*60cm.)

[//]: # (For TA-C, the patch size is 15cm\*15cm.)

### R#26A-2: Performance Against Moving Vehicles.

**1) Speed Limitations:** The attack video under various speeds can be found at [Google Drive](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=drive_link).

**2) Unclear Definitions:** The specific patches are presented in [Figure 1](#r26a-1-unclear-attack-setup).

**3) Laser Operation:** The laser attack operation video can be found at [Google Drive](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=drive_link).

### R#26A-3: Stealthiness of the Attack.

We show our stealthy patches in [Figure 1](#r26a-1-unclear-attack-setup) and the clear comparison in the below table.

| Previous Study | PhysGan[a] (CVPR20) | SLAP[b] (Usenix21) | AoR[c] (Usenix24) | FINE[d] (ICLR24) | IR[e] (NDSS24) | TPatch[f] (Usenix23) | Ours       |
| :------------: |:--------------------|:------------------:|:-----------------:|:----------------:|:--------------:|:--------------------:| :--------: |
| Patch Size     | 122cm×183cm         |     60cm×60cm      |     50cm×50cm     |   200cm×200cm    |   60cm×60cm    |      60cm×60cm       | 60cm×60cm  |

[a] Physgan: Generating Physical World-Resilient Adversarial Examples for Autonomous Driving.\
[b] Slap: Improving Physical Adversarial Examples With Short-Lived Adversarial Perturbations.\
[c] Adversary Is on the Road: Attacks on Visual Slam Using Unnoticeable Adversarial Patch.\
[d] Fusion Is Not Enough: Single Modal Attacks on Fusion Models for 3D Object Detection.\
[e] Invisible Reflections: Leveraging Infrared Laser Reflections to Target Traffic Sign Perception\
[f] Tpatch: A Triggered Physical Adversarial Patch.

## R#26B

### R#26B-1: Design Clarifications.

**1) Patch Definitions:** Clearer definition of patches is show in [Figure 1](#r26a-1-unclear-attack-setup)

## R#26C

### R#26C-1: Stealthiness.

We show the video of driver's perspective in [Google Drive](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=drive_link).
Extra physical evaluation videos in outdoor environment are available in [Google Drive](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=drive_link).

### R#26C-2: More experiments/Discussion Needed.

**2) Speed Limitations:** The attack video under various speeds can be found at [Google Drive](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=drive_link).

**3) Inadequate Countermeasures:** We present the defense results of adversarial training, input-transformation defense, and patch detectors.
mAP (mean Average-Precision) indicate the model's performance.

|          Defence Method          | ASR(Before Defence) | ASR(After Defence) | mAP(Before Defence) | mAP(After Defence) |
|:--------------------------------:| :-----------------: | :----------------: | :-----------------: | :----------------: |
| Adversarial Training[g] (ICCV19) | 94\.4%              | 41\.6%             | 45\.4               | 34\.5              |
| Input-Transformation[h] (CCS17)  | 94\.4%              | 68\.6%             | 45\.4               | 28\.8              |

[g] Towards Adversarially Robust Object Detection\
[h] Magnet: A Two-Pronged Defense Against Adversarial Examples

|       Patch Detector        | Before Defence (without attack) | Before Defence (with attack) | After Defence (without attack) | After Defence (with attack) |
|:---------------------------:| :-----------------------------: | :--------------------------: | :----------------------------: | :-------------------------: |
|     SentiNet[i] (S&P20)     | 100%                            | 85\.1%                       | 97\.5%                         | 0\.6%                       |
|  PatchGuard[j] (Usenix21)   | 100%                            | 96\.9%                       | 95\.3%                         | 0\.0%                       |
| PatchCleanser[k] (Usenix22) | 100%                            | 85\.4%                       | 96\.2%                         | 1\.0%                       |

[i] Sentinet: Detecting Localized Universal Attacks Against Deep Learning Systems\
[j] Patchguard: A Provably Robust Defense Against Adversarial Patches via Small Receptive Fields and Masking\
[k] Patchcleanser: Certifiably Robust Defense Against Adversarial Patches for Any Image Classifier

### R#26C-3: Ethical Responsibilities.

To ensure that our experiments do not cause any harm to the autonomous vehicle camera systems or other related infrastructures, 
and to adhere to ethical and legal standards similar to [31, 32], 
we exercise extreme caution in our research. 
First, all experiments are conducted using our own purchased autonomous vehicle hardware and accounts. 
Second, in line with existing research [33-35], 
we configure appropriate time intervals between laser attacks to avoid disrupting the normal functionality of the camera system and to prevent overloading it during the testing phase. 
Specifically, laser pulses are sent at intervals of 5-10 seconds. 
Lastly, all vulnerabilities identified during the experiments have been promptly reported to the respective vendors, 
and we have received acknowledgments from them. 
For instance, a camera manufacturer has confirmed that the vulnerabilities we discovered in their camera system (#24) were due to firmware issues and have been addressed in the latest firmware update.

## R#26D

### R#26D-1: Ethical Concerns.

We will address ethical concerns in the revised manuscript.

### R#26D-2: Safety Limitations.

Faster speeds evaluation are conducted and the average attack success rate is 56% at 50km/h.
Please see R#26A-2 for details.

### R#26D-3: End-to-End Evaluation.

We conduct end-to-end evaluation on the TurBot3-ARM, an autonomous driving platform.
The platform's pipeline (e.g., data preprocessing algorithms and DNN models) is black-box to us.
Then, we conduct HA, CA, and TA-D against this platform and achieve an average attack success rate of 100%.
The attack demo is available in [LHawk-Lab](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=drive_link).

### R#26D-4: Defense Evaluation.

We test the performance of adversarial training, input-transformation defense, and patch detector.
Although adversarial training and input-transformation defense reduce our attack success rate by an average of 39.3%, the model's performance also decreased by average 30.3%.
Moreover, the patch detector can detect the patch when triggered but fail to detect the patch under benign scenarios, which is due to our triggered adversarial patch design.
Please see R#26C-2 for details.

### R#26D-5: Transferability.

To the best of our knowledge, there is no works to successfully transfer from one model to any other model.
In fact, for transfer attack on a target model, we can always find a model that can achieve a minimum attack success rate of 28.3%.
We will further investigate techniques for improving transferability in our future research efforts.




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
