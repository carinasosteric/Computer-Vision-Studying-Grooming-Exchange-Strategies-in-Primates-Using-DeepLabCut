# Computer-Vision-Studying-Grooming-Exchange-Strategies-in-Primates-Using-DeepLabCut
An end-to-end computer vision pipeline that detects grooming bouts between pairs of animals in video, including who grooms whom. It combines markerless pose estimation, object detection and multi-object tracking, engineered behavioural features, and a frame-level classifier. Rule-based post-processing then turns the frame-level predictions into discrete bouts.

The pipeline takes raw video and pose-estimation output and produces a CSV of predicted bouts per video, with start and end times, duration, direction and confidence. It is designed to replace or support manual behavioural coding of long video datasets.

## Pipeline overview
Video ──► 0. DeepLabCut SuperAnimal pose estimation (multi-animal keypoints, .h5)
            │
            ├─► 1. Self-supervised YOLO training (boxes generated from keypoints)
            │
Video ──► 2. YOLO detection + ByteTrack tracking → track stitching → A/B identity timelines
            │
            ▼
         3. Keypoint-to-identity assignment → interpolation → Kalman smoothing
            │
            ▼
         4. Frame-level labels from manual bout annotations   (training only)
            │
            ▼
         5. Feature extraction (distances, velocities, orientation, confidence, rolling stats)
            │
            ▼
         6. Random Forest classifier: A_gr_B / B_gr_A / no_grooming
            │
            ▼
         7. Post-processing → bout extraction → bout-level evaluation
            │
            ▼
        10. Video quality control → batch inference on the full dataset

## Methodology
### 0. Pose estimation (DeepLabCut)

Keypoints are extracted with DeepLabCut's pretrained SuperAnimal model in multi-animal mode. This needs no species-specific keypoint labelling. Ten keypoints are tracked per animal:


Head: nose, both mouth corners, both ear bases

Body: neck base, mid-back, tail base

Forelimbs: both front paws


DeepLabCut outputs one .h5 file per video, holding x/y coordinates and a likelihood for each keypoint. The rest of the pipeline uses these files for three things: training the detector, assigning keypoints to the two animals, and computing behavioural features.

DeepLabCut's own animal labels aren't stable from frame to frame, especially when animals overlap. That is why the pipeline adds its own detection and tracking stage (step 2) to give each animal a consistent identity.

### 1. Detector training without manual box labels

No bounding boxes were hand-annotated. Training labels for the detector are generated automatically from pose-estimation keypoints:

Box generation: For each animal in each frame, a padded box is built around its keypoints. Frames are sampled at a fixed stride.

Quality filtering: Each candidate box passes plausibility checks that reject keypoints split across overlapping animals:

Head coherence: head keypoints must form a tight cluster.

Spine linearity: neck, mid-back and tail base must be roughly collinear.

Size and aspect ratio: limits reject implausible boxes and small detections such as juveniles.

Keypoint spread: keypoints must not be scattered relative to the box size.


Selection: A weighted quality score combines these checks with keypoint likelihood and count. Only frames with at least two good-quality animals are kept, and the two best boxes per frame are used.
Training: YOLOv8n is fine-tuned on these labels (80/20 train/val split) with augmentation for lighting, scale, rotation and occlusion. Results are checked visually by rendering boxes onto sample frames and videos.

### 2. Tracking and identity assignment

Detection and tracking: The trained detector runs on every frame. Small boxes are filtered out, and ByteTrack links detections into short tracks.

Track stitching: ByteTrack fragments tracks during occlusion and close contact, which is exactly when grooming happens. Fragments are stitched into longer identity chains by linking each track to the best successor that starts shortly after it ends. The cost combines spatial distance and time gap, and a box-size constraint prevents linking animals of different sizes.

Pair selection: The video is split into ~5 s windows. In each window, every pair of chains is scored on co-detection rate and centroid proximity, and the best pair is taken as the interacting dyad.

A/B labelling: The two animals are labelled A and B by horizontal position in the first window. Labels then carry over between windows using chain identity, which avoids swaps when animals cross. Short gaps are filled by linear interpolation of box coordinates.


A and B are consistent within a video. They do not identify specific individuals across videos.

### 3. Keypoint assignment and smoothing

Assignment to A and B: Pose estimation can label the same animal differently from frame to frame. To fix this, each animal's keypoints are assigned to box A or B by how many fall inside each box, falling back to the nearest box centroid. If two keypoint sets claim the same box, the one with higher mean likelihood wins.

Gap filling: Short gaps (≤ 10 frames) are linearly interpolated. Interpolated points keep a likelihood of 0, so later stages know they are estimates.

Smoothing: A per-keypoint constant-velocity Kalman filter smooths each trajectory. Its measurement noise scales inversely with the pose-estimation likelihood, so confident detections pull the estimate more strongly.

### 4. Training labels

Manual annotations give the start, end, groomer and recipient for each bout. These are expanded into one label per frame: A_gr_B, B_gr_A or no_grooming.

### 5. Feature engineering

Features are computed for every frame:

Inter-animal distance:	Each paw of one animal to the other's nose, neck, back and tail base; centroid distance; minimum paw-to-body distance per direction

Confidence:	Likelihood product for each distance pair; mean keypoint likelihood per animal; A/B likelihood ratio

Motion:	Paw, nose and body velocity per animal; relative body velocity

Orientation	Facing angle between animals; body axis angle per animal; body angle difference

Detection flags:	Whether A, B, both or neither were genuinely detected rather than interpolated

Temporal context:	Rolling mean and standard deviation over short (0.2 s) and long (1 s) centred windows


The direction-specific paw-to-body features (A→B vs B→A) are what let the classifier tell who is grooming whom.

### 6. Classification

Model: A Random Forest (1,000 trees, balanced class weights, min_samples_leaf=20) inside a scikit-learn Pipeline with median imputation.

Validation: Group K-fold cross-validation grouped by video, so frames from one video never appear in both train and test folds. Because neighbouring frames are highly correlated, this avoids the inflated scores a random frame-level split would give. Imputation is fitted within each fold to prevent leakage.

Outputs: A classification report, confusion matrix and feature importances.

### 7. Post-processing into bouts

Frame-level predictions are noisy, so they are turned into bouts in stages:

Direction smoothing: a rolling mean of each direction's probability over 10 frames.

Thresholding: a frame counts as grooming if P(grooming) > 0.6.

Short-bout removal: bouts under 1 s are dropped.


Gap merging:

Gaps under 0.5 s are merged regardless of direction (signal dropout).

Gaps under 2 s are merged between same-direction bouts. A later pass extends this to 5 s.

Isolated-bout removal: short bouts (< 3 s) more than 30 s after the previous bout are dropped as likely false positives.


Each bout records its mean probabilities and a direction confidence, defined as |P(A→B) − P(B→A)|.


Bout-level evaluation:

Predicted bouts are compared against annotated bouts. A true bout counts as detected if a predicted bout overlaps at least 50% of it. Three metrics are reported:

Bout count accuracy: per video

Duration error: absolute and percentage, for matched bouts

Direction accuracy: for matched bouts

8–12. Inference on the full dataset


The trained detector and classifier run on the full video set using the same tracking, assignment and feature steps.

Before classification, each video gets a quality score. It is based mostly on how often both animals have confident keypoints, along with per-animal coverage, mean pose-estimation likelihood and track stability (median frame-to-frame centroid jump). A video is kept only if its overall quality score is at least 0.3 and both animals have confident keypoints in at least 10% of frames. Videos that fail either test are excluded before classification.

The final outputs are per-frame class probabilities and per-video bout predictions.

## Requirements
Python 3.10+ (developed on 3.13)

A CUDA GPU is recommended for YOLO training and inference

DeepLabCut with the pretrained SuperAnimal model, for pose estimation. Run it separately; this pipeline reads its .h5 output.
bash

pip install ultralytics supervision opencv-python pandas numpy scikit-learn matplotlib pyyaml tables


Inputs:


Videos: .mp4, 960×540, 25 fps

Pose estimation: Multi-animal DeepLabCut SuperAnimal .h5, one per video (filenames contain _superanimal), with the keypoints nose, mouth_end_left/right, left/right_earbase, neck_base, back_middle, tail_base, front_left/right_paw

Annotations (training only): CSV with one row per video and columns bout_{i}_start, bout_{i}_end, bout_{i}_gr, bout_{i}_rec (groomer and recipient as A/B)

## Usage

The notebook runs top to bottom in two phases:

Training (sections 1–7): train the detector, build A/B tracks for the annotated videos, extract features, train and cross-validate the classifier, then tune the post-processing.

Inference (sections 8–12): run tracking, assignment, quality control, feature extraction and classification on the full video set.

Set the folder paths in each section's settings cell before running.

## Outputs

*_boxes.csv: per-frame A/B bounding boxes

*_cleaned.csv: per-frame A/B keypoints after assignment and smoothing

all_frame_predictions.csv: per-frame class probabilities

all_bout_predictions.csv: predicted bouts with timing, direction and confidence

## Acknowledgements
- DeepLabCut and the SuperAnimal pretrained models (Ye et al., 2024, Nature Communications): pose estimation
- Ultralytics YOLOv8: object detection
- ByteTrack via supervision: multi-object tracking
- scikit-learn: Random Forest classification
