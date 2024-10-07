# Optimized Road Damage Detection Challenge (ORDDC`2024)

[The Optimized Road Damage Detection Challenge](https://orddc2024.sekilab.global/overview/) addresses the problem of automating the road damage detection (RDD) targeting the optimized inference speed and resource usage. So far, the RDD challenges have prioritized enhancing the performance of RDD algorithms/models, with the F1-score serving as the primary (and sole) metric. However, moving forward, it has become increasingly crucial to address resource optimization concerns, particularly regarding inference speed and memory usage, to enable real-time deployment of these models. Therefore, the current challenge shifts the primary criterion towards optimizing resource usage. The background details are provided as follows.

---

## Phase 1
- **Folder:** `/phase1`

The results of the model proposed by the participants are evaluated by "F-Measure." The prediction is correct only when IoU (Intersection over Union, see Fig.1) is 0.5 or more, and the predicted label and the ground truth label match. Dividing the area of overlap by the area of the union yields intersection over union

### Installation
To install MMDetection, please refer to the official [documentation](https://mmdetection.readthedocs.io/en/latest/install.html).

### Models
The following models are implemented:
- **Model 1:** Faster-RCNN + ConvNextv2 (base) [[download](https://drive.google.com/drive/folders/1BD1RhV-9AllfNw75LvHne6cc1n_di0wr?usp=sharing)]
- **Model 2:** Faster-RCNN + SwinTransformer (tiny) [[download](https://drive.google.com/drive/folders/1BD1RhV-9AllfNw75LvHne6cc1n_di0wr?usp=sharing)]
- **Model 3:** Faster-RCNN + SwinTransformer (large) [[download](https://drive.google.com/drive/folders/1BD1RhV-9AllfNw75LvHne6cc1n_di0wr?usp=sharing)]
- **Model 4:** Faster-RCNN + SwinTransformer (small) [[download](https://drive.google.com/drive/folders/1BD1RhV-9AllfNw75LvHne6cc1n_di0wr?usp=sharing)]
- **Model 5:** Faster-RCNN + SwinTransformer (small) [[download](https://drive.google.com/drive/folders/1BD1RhV-9AllfNw75LvHne6cc1n_di0wr?usp=sharing)]

### Inference
For running inference in Phase 1, utilize the provided Jupyter notebook:
- **File:** `inference.ipynb`

### LeaderBoard
- F1-Score: `0.7165`

---

## Phase 2
- **Folder:** `/phase2`

Inference Speed of the models proposed by the participants would be the primary evaluation criteria. Participants need to submit their proposed model file (saved checkpoint, .pb file or some other format), along with the inference script (.py file) and corresponding information of implementation requirements (.txt). Update (17/08/2024): In case of two teams having large difference between one metric (F1-score and Inference Speed) and comparable performance for other one, the rank will be decided based on the submitted report and reviewers comments.

### Installation
To install the Ultralytics framework, use the following command:
```bash
pip install -U ultralytics
```

### Model
- **Model 1:** Yolov9e [[download](https://drive.google.com/drive/folders/1BD1RhV-9AllfNw75LvHne6cc1n_di0wr?usp=sharing)]

### Inference
For inference in this phase, use the provided Python script:
- **File:** `inference.py`

### LeaderBoard
- F1-Score: `0.6202`
- Inference Speed: `0.0782 (sec/image)`

---

## Contribution
The project welcomes contributions to improve the models and methodologies used in this challenge. Interested developers are encouraged to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License, allowing open use and modification.

## Acknowledgments 
Special thanks to all contributors and the community for supporting the success of this challenge.