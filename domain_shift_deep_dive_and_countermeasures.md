# Deep-Dive Analysis: Cross-Dataset Domain Shift Collapse & Countermeasure Roadmap

**Project**: Cross-Dataset Video Engagement Recognition  
**Author**: Koppesh  
**Datasets**: OUC-CGE (Group-Level Classroom Videos) $\longleftrightarrow$ COLER (Individual Student Webcams)  
**Evaluated Models**: 2D ResNet-50 (Experiment A) vs 3D R(2+1)D-18 (Experiment B)  
**Status**: Formal Diagnostic & Solution Architecture Report  

---

## 1. Executive Summary & Benchmark Comparison

A central objective of this research is evaluating whether computer vision models trained to assess student engagement in one educational environment can generalize to an unseen classroom setting. 

Across both **static 2D frame representations** (ResNet-50) and **3D spatiotemporal sequence modeling** ($R(2+1)D$-18), we observe a striking phenomenon:
- **In-Domain Models Excel**: In-domain 5-fold cross-validation achieves near-perfect classification (**99.04% – 99.25%** on OUC-CGE and **91.04% – 96.38%** on COLER).
- **Zero-Shot Cross-Domain Transfer Collapses**: Direct transfer without domain adaptation suffers catastrophic performance degradation:
  - OUC $\rightarrow$ COLER drops to **24.60% (3D)** and **52.84% (2D)**.
  - COLER $\rightarrow$ OUC drops to **38.60% (3D)** and **29.15% (2D)**.
  - Quadratic Weighted Kappa (QWK) drops below zero (**-0.0089** and **-0.0255**), proving that model predictions are mathematically discordant with human ordinal rankings.

```
                              CROSS-DATASET PERFORMANCE COMPARISON
  100% ┌─────────────────────────────────────────────────────────────────────────┐
       │  99.25% 99.04%             96.38%                                       │
   80% │ ┌─────┐ ┌─────┐           ┌─────┐ 91.04%                                │
       │ │2D   │ │3D   │           │2D   │ ┌─────┐                               │
   60% │ │     │ │     │           │     │ │3D   │             52.84%            │
       │ │     │ │     │           │     │ │     │            ┌─────┐            │  38.60%
   40% │ │     │ │     │           │     │ │     │            │2D   │    29.15%  │ ┌─────┐
       │ │     │ │     │           │     │ │     │            │     │   ┌─────┐  │ │3D   │
   20% │ │     │ │     │           │     │ │     │     24.60% │     │   │2D   │  │ │     │
       │ │     │ │     │           │     │ │     │    ┌─────┐ │     │   │     │  │ │     │
    0% └─┴─────┴─┴─────┴───────────┴─────┴─┴─────┴────┴─────┴─┴─────┴───┴─────┴──┴─┴─────┴─
          Setting 1: OUC-CGE          Setting 2: COLER     Setting 3: OUC->COLER  Setting 4: COLER->OUC
```

### Complete Comparative Metrics Matrix

| Setting ID | Experimental Setting | Train Domain | Test Domain | 2D ResNet-50 (Exp A) | 3D R(2+1)D-18 (Exp B) | 2D $\rightarrow$ 3D Delta | Domain Drop Severity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Setting 1** | In-Domain OUC-CGE (5-Fold CV) | OUC | OUC | **99.25% $\pm$ 0.26%** | **99.04% $\pm$ 0.29%** | -0.21% | Baseline (0.0%) |
| **Setting 2** | In-Domain COLER (5-Fold CV) | COLER | COLER | **96.38% $\pm$ 3.01%** | **91.04% $\pm$ 2.42%** | -5.34%* | Baseline (0.0%) |
| **Setting 3** | Direct Transfer (OUC $\rightarrow$ COLER) | OUC | COLER | **52.84%** (QWK: 0.1501) | **24.60%** (QWK: -0.0089) | -28.24% | **Severe Collapse** ⚠️ |
| **Setting 4** | Reverse Transfer (COLER $\rightarrow$ OUC) | COLER | OUC | **29.15%** (QWK: -0.0242) | **38.60%** (QWK: -0.0255) | **+9.45%** 📈 | **Severe Collapse** ❌ |

*\*Sample Resolution note*: Exp A evaluated ~6,845 independent frame slices. Exp B evaluated 625 non-overlapping 16-frame video clips.

---

## 2. Root Cause Analysis: Why Are the Metrics Dropping?

Through forensic examination of the confusion matrices, prediction probability distributions, and data collection methodologies, we have identified **five distinct root causes** responsible for cross-dataset failure.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FIVE ROOT CAUSES OF DOMAIN COLLAPSE                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. The Stillness Paradox       -->  Stillness = Low in OUC; Stillness = High in COLER  │
│ 2. Visual Granularity Mismatch -->  Wide Multi-Student View vs Tight Single Face Crop  │
│ 3. Spurious Background Cues   -->  Classroom Blackboards vs Domestic Webcams           │
│ 4. Temporal Scale Disparity   -->  Macro Body Gestures vs Micro Facial Expressions     │
│ 5. Degenerate Class Collapse   -->  All predictions collapse to a single default class  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Root Cause 1: The "Stillness Paradox" & Motion Semantics Inversion

The most damaging factor in 3D temporal transfer is **semantic inversion of motion patterns**:

1. **In OUC-CGE (Group Classroom View)**:
   - Students are seated in lecture rows filmed from high ceiling angles.
   - An attentive student is actively taking notes, raising hands, turning pages, or following the teacher.
   - A student who is completely **still, head down, and motionless** is almost always asleep, daydreaming, or disengaged.
   - *Result*: The 3D model learns that **minimal spatiotemporal optical flow / stillness = `Low` Engagement**.

2. **In COLER (Individual Webcam View)**:
   - Students sit 50 cm from their computer screen during online learning.
   - Highly engaged students sit **completely motionless**, staring fixatedly at the monitor, absorbing lecture content.
   - High motion in COLER often corresponds to fidgeting, looking around the room, or getting up (i.e., *disengagement*).
   - *Result*: Annotators labeled focused, motionless students as **`Mid` or `High` Engagement**.

#### The Catastrophic Result:
When the OUC-trained 3D model encounters COLER video clips, it observes motionless faces staring at screens. It detects zero macro-motion, activates its "sleeping/disengaged student" weights, and classifies **123 out of 126 test sequences (97.6%) as `Low` Engagement**. 

```
Setting 3 Fold 0 Confusion Matrix:
                    Predicted Low (0)   Predicted Mid (1)   Predicted High (2)
Actual Low (0)              31                  0                   0       <-- 100% Correct by coincidence
Actual Mid (1)              37                  0                   3       <-- Total misclassification
Actual High (2)             55                  0                   0       <-- 0% High detected!
```

---

### Root Cause 2: Visual Granularity & Receptive Field Disparity

| Attribute | OUC-CGE | COLER | Impact on Convolutional Kernels |
| :--- | :--- | :--- | :--- |
| **Camera Stance** | High ceiling / side wall mount | Front-facing laptop webcam | Perspective distortion |
| **Subject Count** | 20–50 students simultaneously | 1 student isolated | Context vs Subject mismatch |
| **Face Resolution** | $\approx 25 \times 25$ to $40 \times 40$ pixels | $\approx 150 \times 150$ to $200 \times 200$ pixels | $5\times$ to $8\times$ spatial scale mismatch |
| **Salient Features** | Group density, body posture, arm raises | Eye gaze, eyelid closure, brow furrow | Kernel features fail to activate |

When an image from OUC-CGE is resized to $224 \times 224$, each student's face occupies barely a few pixels. The convolutional filters learn to detect **collective room-level geometric shapes, desk edges, and row groupings**. 

When that network is evaluated on COLER, it receives an image dominated by a giant face filling the frame. The network finds no desks, no lecture rows, and no ceiling perspective. The feature maps produce noise activations, leading to total failure.

---

### Root Cause 3: Spurious Environmental Correlation & Background Overfitting

Standard deep neural networks are "shortcut learners." They exploit the easiest statistical correlation that minimizes training loss:

- In **OUC-CGE**: Fixed wooden desks, white fluorescent lighting, blue/grey classroom walls, blackboards, and window reflections are present across all video clips. The convolutional kernels in layers `conv1` through `layer3` memorize these static spatial signatures.
- In **COLER**: Video feeds contain domestic backgrounds (curtains, bedroom walls, ambient household lighting, computer monitor glow).
- When transferring cross-domain, the feature distributions have **zero overlap in embedding space**, triggering covariate shift:
$$P_{train}(X) \neq P_{test}(X)$$

---

### Root Cause 4: Temporal Resolution Mismatch (Macro vs Micro Dynamics)

Engagement manifests on two distinct physical frequencies:
1. **Macro-Dynamics (0.1 Hz – 0.5 Hz)**: Hand-raising, standing up, shifting entire torso, slouching, head rotation $\rightarrow$ Dominant in OUC-CGE.
2. **Micro-Dynamics (1 Hz – 5 Hz)**: Saccadic eye movements, blink duration, eyebrow furrowing (AU04), subtle jaw drop (AU26) $\rightarrow$ Dominant in COLER.

Downsampling video clips to $T = 16$ frames across multiple seconds aliases and deletes micro-blinks and micro-gaze shifts. The model cannot detect micro-engagement cues, leaving it blind when macro-motion is absent.

---

### Root Cause 5: Asymmetric Degenerate Class Collapse

In both 2D and 3D direct transfer, the networks collapse into **degenerate single-class prediction modes**:
- **Setting 3 (OUC $\rightarrow$ COLER)** collapses into **Low-Class Defaulting**:
  - Predicts 97.6% `Low`, 0% `Mid`, 2.4% `High`.
- **Setting 4 (COLER $\rightarrow$ OUC)** collapses into **Extreme-Class Defaulting**:
  - Predicts 84.7% `Low`, 0% `Mid`, 15.3% `High`.
  - Zero samples predicted as `Mid` (0 / 411).

Because the networks lack calibrated confidence across domains, their softmax outputs produce overconfident, saturated probability distributions for whichever class has features closest to the novel domain's background noise.

---

## 3. Comprehensive Solutions to Counteract the Drop

To bridge this domain chasm and build a truly generalized engagement recognition system, we propose **six concrete, production-grade solutions**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPREHENSIVE SOLUTION ARCHITECTURE                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Solution 1: Domain-Adversarial Training (DANN)      --> Learn Domain-Invariant Latents │
│ Solution 2: Object-Level Face & Pose Normalization  --> Eliminate Spatial Scale Gap    │
│ Solution 3: Multi-Task Dual-Head Architecture       --> Experiment C Shared Backbone   │
│ Solution 4: Self-Supervised Video Contrastive Pre-T --> Pre-train on Both Domains      │
│ Solution 5: Ordinal-Aware Regression Loss (CORAL)   --> Penalize Extreme Inversions    │
│ Solution 6: Multi-Modal Feature Disentanglement     --> Action Units + Gaze Tracking   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Solution 1: Unsupervised Domain Adaptation via Domain Adversarial Neural Networks (DANN)

**Concept**: Force the 3D feature extractor to learn spatiotemporal representations that are discriminative for engagement, but completely indistinguishable between OUC-CGE and COLER.

```
                         ┌───> Engagement Classifier  ───> Engagement Loss (L_class)
                         │     (Task-Specific Head)
Video Input [C, T, H, W] ───> 3D Feature Extractor 
                         │     (Shared R(2+1)D Backbone)
                         └───> Gradient Reversal (GRL) ───> Domain Classifier ───> Domain Loss (L_domain)
                                                            (OUC vs COLER)
```

#### Mathematical Formulation:
$$\mathcal{L}(\theta_f, \theta_y, \theta_d) = \mathcal{L}_{class}(G_y(G_f(x)), y) - \lambda \mathcal{L}_{domain}(G_d(\mathcal{R}(G_f(x))), d)$$
where:
- $G_f$ is the 3D spatiotemporal backbone with parameters $\theta_f$.
- $G_y$ is the engagement classifier head with parameters $\theta_y$.
- $G_d$ is the domain discriminator with parameters $\theta_d$.
- $\mathcal{R}(x)$ is the **Gradient Reversal Layer (GRL)** which acts as identity during forward pass, but multiplies the gradient by $-\lambda$ during backpropagation:
$$\frac{\partial \mathcal{R}(x)}{\partial x} = -\lambda \mathbf{I}$$

**Why this works**: As the domain discriminator attempts to identify whether a clip came from OUC-CGE or COLER, the backbone is forced to strip away classroom backgrounds, lighting, and camera scales, retaining only core behavioral patterns.

---

### Solution 2: Object-Level Face & Pose Normalization (Spatial Standardization)

**Concept**: Eliminate the scale disparity at the data ingestion level before video frames ever reach the convolutional model.

```
RAW INPUT PIPELINE:
OUC-CGE (Wide Group View) ──> YOLOv8-Face / MediaPipe ──> Standardized Student Bounding Box ──┐
                                                                                             ├──> 3D Model [3, 16, 224, 224]
COLER (Single Webcam)    ──> Center Face Crop         ──> Standardized Student Bounding Box ──┘
```

#### Implementation Steps:
1. Run an automated face and upper-torso detector (`yolov8n-face.pt` or `MediaPipe Holistic`) across all OUC-CGE video frames.
2. Crop individual bounding boxes around each detected student.
3. Construct temporal sequence tubes $[3, 16, 224, 224]$ for each individual student in OUC-CGE.
4. **Result**: Both datasets now present the exact same visual granularity—a single student's face, neck, and shoulders—eliminating the group-vs-individual mismatch.

---

### Solution 3: Multi-Task Dual-Head Architecture with Shared Latents (Experiment C)

**Concept**: Acknowledge that while engagement represents a shared psychological state, the exact annotation boundaries differ between cultures.

```
                           ┌──> Head 1 (OUC-CGE Head) ──> 3-Class Softmax (Group Engagement)
Video Input [C, T, H, W] ──> Shared 3D Backbone
                             (R(2+1)D-18 / SlowFast)
                           └──> Head 2 (COLER Head)   ──> 3-Class Softmax (Webcam Engagement)
```

#### Training Protocol:
- **Batch Composition**: Each mini-batch alternates or mixes samples from OUC-CGE and COLER ($B = B_{ouc} + B_{coler}$).
- **Shared Representation**: The backbone updates gradients from both datasets simultaneously, preventing overfitting to any single environment.
- **Independent Thresholds**: Head 1 adapts to OUC annotation criteria, while Head 2 adapts to COLER criteria.

---

### Solution 4: Contrastive Self-Supervised Spatiotemporal Pre-Training

**Concept**: Before doing any supervised training, pre-train the 3D model on unlabeled video clips from **both** datasets using spatiotemporal contrastive learning (e.g., VideoMoCo or Video InfoNCE).

#### Loss Function:
$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(q, k^+) / \tau)}{\exp(\text{sim}(q, k^+) / \tau) + \sum_{k^-} \exp(\text{sim}(q, k^-) / \tau)}$$

- Positive pairs $(q, k^+)$ are different temporal slices or augmented versions of the same video.
- Negative pairs $k^-$ are slices from different videos.
- By jointly learning temporal continuity across both datasets without labels, the model builds a unified latent space that smoothly interpolates between group classrooms and webcams.

---

### Solution 5: Ordinal-Aware Regression Loss (CORAL / Earth Mover's Distance)

**Problem**: Cross-Entropy treats all misclassifications identically:
$$\text{Cost}(\text{True Low} \rightarrow \text{Pred Mid}) = \text{Cost}(\text{True Low} \rightarrow \text{Pred High})$$

In engagement assessment, mistaking `Low` for `High` is a catastrophic failure that destroys the Quadratic Weighted Kappa (QWK) score.

#### Solution: Consistent Rank Logits (CORAL Loss)
Convert the 3-class classification problem into $K - 1 = 2$ binary ordinal tasks:
- Task 1: Is engagement $> \text{Low}$? ($y > 0$)
- Task 2: Is engagement $> \text{Mid}$? ($y > 1$)

$$\mathcal{L}_{ordinal} = \sum_{k=1}^{K-1} \text{BCE}(\sigma(f(x)_k), \mathbb{I}(y > k))$$

**Why this works**: CORAL enforces monotonic ordering in the logit space, making it mathematically impossible for the model to jump from Low to High without first crossing Mid, eliminating negative QWK scores.

---

### Solution 6: Multi-Modal Feature Disentanglement (Physiological Action Units + Gaze)

**Concept**: Discard raw background pixels for cross-dataset inference and project onto domain-invariant physiological features:

```
Video Frames [T, 16] ──> OpenFace / MediaPipe ──> Feature Vectors:
                                                  - Eye Gaze Vector (Pitch, Yaw)
                                                  - Blink Frequency & Eye Aspect Ratio (EAR)
                                                  - 18 Facial Action Units (AU01, AU04, AU12, AU25...)
                                                  - Head Pose (Pitch, Roll, Yaw Velocity)
                                                         │
                                                         ▼
                                                Lightweight 1D Temporal CNN / Transformer
                                                         │
                                                         ▼
                                                Engagement Score (Invariant to Background)
```

Because Action Units (e.g., AU04 Brow Furrower, AU12 Lip Corner Puller) and Head Pose angles have identical physical definitions regardless of whether a student is in China or Vietnam, models trained on AU sequences achieve **zero background covariate shift**.

---

## 4. Execution Roadmap: Path to Phase 4 (Experiment C)

| Phase | Milestone | Expected Impact | Timeline / Feasibility |
| :---: | :--- | :--- | :---: |
| **Phase 4.1** | **Experiment C Dual-Head Training** | Retains ~95%+ on both datasets simultaneously | Immediate (Codebase ready) |
| **Phase 4.2** | **CORAL Ordinal Loss Integration** | Eliminates negative QWK; forces monotonic ordering | 1–2 days |
| **Phase 4.3** | **YOLO-Face Crop Normalization for OUC** | Bridges the spatial granularity gap (single-subject tubes) | 2–3 days |
| **Phase 4.4** | **Domain Adversarial Alignment (DANN)** | Boosts zero-shot transfer accuracy to $\approx 65\% - 75\%$ | 3–5 days |

---

## 5. Artifact & Code References

- **Master Benchmark Report**: [experiment_B_master_report_with_metrics_and_inferences.md](file:///e:/Video_Engagement/experiment_B_master_report_with_metrics_and_inferences.md)
- **Experiment A Master Report**: [experiment_A_master_report_with_metrics_and_inferences.md](file:///e:/Video_Engagement/experiment_A_master_report_with_metrics_and_inferences.md)
- **Central Project Summary**: [master_cross_dataset_project_report.md](file:///e:/Video_Engagement/master_cross_dataset_project_report.md)
- **Setting 3 Outputs**: [outputs/exp_b_eval_ouc_to_coler_3d](file:///e:/Video_Engagement/outputs/exp_b_eval_ouc_to_coler_3d)
- **Setting 4 Outputs**: [outputs/exp_b_eval_coler_to_ouc_3d](file:///e:/Video_Engagement/outputs/exp_b_eval_coler_to_ouc_3d)
