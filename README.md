This repository is related to the article: "Comparing torsional and lateral vibration data for deep learning based drive train gear
diagnosis". The article has been submitted on the 23rd August, 2022.

The related dataset will be published alongside the article. The dataset contains artificially produced gear faults, measured with numerous accelerometers, torque transducers and rotary encoders at multiple rotating speeds between 250 RPM and 1500 RPM.

The models were optimised with the following hyperparameters:

| Hyperparameter | WDCNN | SRDCNN | Ince |
| :---  | :--- | :--- | :--- |
| Batch size | 64 | 64 | 32 |
| Learning rate| 0.001 | 0.0001 | 0.001 |
| Num epochs | 20 | 20 | 20 |
| Time window length | 2048 | 2048 | 2048 |
| Time window stride | 32 | 32 | 32 |
| Early stop patience epochs | 10 | 10 | 10 |

To reproduce the results:

- Load data from: [Dataset will be published with the article]

- cd ./code
- python3 -m venv env
- source env/bin/activate
- pip install -r requirements.txt

- python3 main.py --dataset traditional --arch WDCNN --lr 0.001 --batch-size 64 --epochs 20 --tw-stride 128 --tw-len 2048

- python3 main.py --dataset traditional --arch SRDCNN --lr 0.0001 --batch-size 64 --epochs 20 --tw-stride 128 --tw-len 2048

- python3 main.py --dataset traditional --arch Ince --lr 0.001 --batch-size 32 --epochs 20 --tw-stride 128 --tw-len 2048 --bias False
