Here is the complete study guide formatted in clean, standard Markdown. You can easily copy and paste this text directly into any Markdown reader or text editor (like Obsidian, VS Code, or Typora), or export it directly to a `.md` file for offline reading.

---

# Study Guide: Deep Dive into the COLER Dataset and the CORE-Net Deep Learning Architecture

## Section 1: Overview and the Core Scientific Problem

In computer vision and machine learning, recognizing human behavior is a core challenge. Most existing systems are trained on datasets recorded in highly controlled, artificial laboratory environments.

When you deploy those models in a real classroom, they fail. Real classrooms—specifically those in developing regions like Vietnam—are crowded, noisy, and dynamic. Students turn their heads, laptops or desks block the camera’s view (occlusions), and their spatial arrangements shift constantly.

Furthermore, most older architectures evaluate students as isolated entities. But in collaborative learning, student engagement is fundamentally social and contextual; individual behavior is heavily influenced by peers, nearby learning materials (like books or phones), and the overall group dynamic.

To solve these issues, the researchers proposed:

1. **The COLER Dataset**: Collaborative Learning Engagement Recognition.

2. **The CORE-Net Architecture**: Contextual Ordinal Engagement Network.

---

## Section 2: The COLER Dataset Structure

To capture authentic, "in-the-wild" classroom dynamics, the authors recorded 70 student volunteers (46 males, 24 females) divided into 30 independent collaborative groups.

### 1. Environmental and Seating Configurations

To ensure the models learned generalizable spatial features rather than memorizing a single seating pattern, the dataset was recorded across three distinct physical table layouts :

- **Cluster Desks**: Desks clustered together in tight blocks to promote face-to-face interaction, resulting in high lateral body occlusions.

- **Traditional Desks**: Standard linear rows and columns, creating high occlusion of lower limbs and hands behind desks.

- **Hexagonal Tables**: Distributed seating arrangements with wide-angle perspectives.

The groups ranged in size from two to six members, and students were allowed to freely swap seats during sessions.

### 2. The Two-Stage Activity Protocol

To naturally elicit different levels of engagement, the sessions were split into two chronological phases :

- **Spontaneous Discussion Stage**: Groups discussed casual, self-selected topics to generate uninhibited, natural social expressions.

- **Collaborative Problem-Solving Stage**: Groups worked together on course-related questions of incremental difficulty, forcing them to consult physical textbooks, debate, write notes, and actively collaborate.

### 3. Preprocessing and Tracking Pipeline

The raw video footage was recorded in HD 720p at 30 frames per second (fps) and segmented into short 5-second clips. The authors implemented a tracking and preprocessing pipeline to select representative frames :

- **Object Detection & Tracking**: Individual students were localized and tracked across time using the **YOLOv11** object detector combined with the **BoT-SORT** tracking algorithm.

- **Temporal Uniform Sampling**: For each tracked student, exactly `$T=15$` frames were uniformly sampled across the 5-second window.

- **Occlusion Recovery**: If a student's face was temporarily blocked or turned away in a sampled frame, the pipeline dynamically replaced it with the chronologically closest frame where their face was visible.

### 4. Ground-Truth Labels & Annotation Statistics

To establish reliable, bias-free labels, the researchers worked with child psychology experts to create a behavioral-emotional expression calibration framework, which was validated via a survey of 498 professional educators. This mapped combinations of expressions and postures into four progressive ordinal engagement levels :

1. **Disengaged**

2. **Normal**

3. **Engaged**

4. **Very Engaged**

Each clip was manually reviewed and labeled by 5 independent trained annotators, with the final label decided by majority vote.

- **Total Individual-Level Samples**: 3,924 annotated clips.

- _Disengaged_: 1,162 samples.

- _Normal_: 807 samples.

- _Engaged_: 1,052 samples.

- _Very Engaged_: 903 samples.

- **Total Group-Level Samples**: 740 annotated clips (which evaluate group-level interactions, speaking frequency, and overall synchrony).

---

## Section 3: Component Analysis of the CORE-Net Architecture

The CORE-Net framework is a multimodal deep learning pipeline. It extracts features from four different parallel branches, projects them into a unified 512-dimensional vector space, models group relationships, fuses the features using attention, and classifies them using an ordinal loss function.

### 1. Modality Branch 1: Pose Feature Extractor (`$F_G\in\mathbb{R}^{512}$`)

CORE-Net employs **BlockGCN** to aggregate joint data into anatomically coherent blocks :

- It groups the 13 physical joints (nose, eyes, ears, shoulders, elbows, wrists, hips) into `$B$` anatomical blocks using a learnable assignment matrix `$S\in\mathbb{R}^{J\times B}$`.

- It calculates block-level features `$Z$` using a **block-attention mechanism** :
  `$$Z=VS\operatorname{softmax}\left(\frac{(QS)(K S)^{\top}}{\sqrt{d}}\right)$$`
  Where `$Q$`, `$K$`, and `$V$` represent the query, key, and value projections of joint coordinate features, and `$d$` is the scaling factor.

- These localized block features are mapped back to joint coordinates via `$S^{\top}$`. Temporal dynamics are captured using multi-scale 1D temporal convolutions, outputting a posture vector `$F_G\in\mathbb{R}^{512}$`.

### 2. Modality Branch 2: Behavioral Dynamic Encoder (`$F_B\in\mathbb{R}^{512}$`)

To capture physical actions like writing, reading, standing up, or turning, CORE-Net processes the raw RGB video sequence through a **MoViNet** backbone.

- The raw spatial-temporal features are aggregated via Global Average Pooling into a 2048-dimensional vector, which is linearly projected down to `$F_B\in\mathbb{R}^{512}$`.

### 3. Modality Branch 3: Facial Emotion Encoder (`$F_E\in\mathbb{R}^{512}$`)

To capture the expressions of the student, a dedicated branch processes the sequence of facial crops. By tracking facial landmark movements over the uniformly sampled frames, this encoder maps affective emotional shifts into a robust expression vector `$F_E\in\mathbb{R}^{512}$`.

### 4. Modality Branch 4: Context Encoder with Instance Segmentation

This branch extracts three global classroom-view frames at the start, middle, and end of the clip: `$\{C_1, C_{T/2}, C_T\}$`.

- **Object & Human Detection**: **Faster R-CNN** localizes human coordinates, and **OWL-ViT** detects physical objects in the room (laptops, phones, books, paper).

- **Pixel-Level Segmentation**: The **Segment Anything Model (SAM)** creates masks `$M_i$` for every detected student and object.

- **Masked Pooling**: It extracts localized, context-rich representations using mask pooling :
  `$$f^i = \operatorname{ResNet}(C_t \odot M_i), \quad i=1,\dots,N$$`
  Where `$\odot$` is a pixel-wise multiplication operator that isolates the student or object from the background noise.

- **Global Regularizer**: Background pixels are suppressed to create a global foreground image `$C_{fg}$`, processed by a Vision Transformer (`$\text{ViT-B/16}$`) to output a global scene descriptor `$F_C = \operatorname{ViT}(C_{fg})\in\mathbb{R}^{512}$`.

- All individual student features `$f^{stu}$`, physical object features `$f^{obj}$`, and the global scene descriptor `$F_C$` are grouped into a **unified context token set** :
  `$$\mathcal{T}_C = \{f_1^{stu}, \dots, f_{N_s}^{stu}, f_1^{obj}, \dots, f_{N_o}^{obj}, F_C\}$$`

---

## Section 4: Relational Group Interaction via Set Transformers

To resolve individual behavior ambiguity using group-level cues, CORE-Net processes the token set `$\mathcal{T}_C$` through a **Set Transformer**.

### 1. Intra-Frame Interaction

For each frame, the tokens exchange behavioral cues using multi-head self-attention :
`$$H_t = \operatorname{softmax}\left(\frac{Q_t K_t^{\top}}{\sqrt{d}}\right) V_t W$$`
This allows student tokens to condition their representations on what other peers are doing, and allows object tokens to integrate student-interaction cues, grounding human behavior directly in the physical environment.

### 2. Pooling by Attention

To aggregate the set of refined tokens `$\mathcal{T}_C' = \{v_j\}_{j=1}^{m}$` into a single frame descriptor `$z_t$`, CORE-Net uses attention-pooling. It introduces a learnable query slot `$S$` conditioned on the global scene descriptor `$F_C$`, directing the attention matrix toward active students and ignoring background noise :
`$$z_i = \sum_{j=1}^{m} \operatorname{softmax}\left(\frac{(s_i W_Q)(v_j W_K)^{\top}}{\sqrt{d}}\right) (v_j W_V)$$`

### 3. Temporal Modeling

The resulting frame descriptors `$\{z_1, z_{T/2}, z_T\}$` are passed through a temporal Transformer encoder with positional encodings `$P$` to model temporal context :
`$$Z_t = \operatorname{TransformerEncoder}( + P)$$`
Averaging across the temporal dimensions yields the final relational context embedding `$F_{CTX}\in\mathbb{R}^{512}$`.

---

## Section 5: Hierarchical Multimodal Attention Fusion

CORE-Net implements a hierarchical two-stage fusion module :

### Stage 1: Intra-Individual Self-Attention

The model constructs a token sequence of the student’s own behavioral modalities `$T_{BE} =\in\mathbb{R}^{3\times d}$`. It applies self-attention to capture cross-modal correlation, dynamically weighting the most reliable cues :
`$$A_{BE} = \operatorname{softmax}\left(\frac{(T_{BE} W_Q)(T_{BE} W_K)^{\top}}{\sqrt{d}}\right) (T_{BE} W_V)$$`
`$$v_{BE} = \operatorname{MeanPool}(T_{BE} + A_{BE})$$`

### Stage 2: Context-Aware Cross-Attention

The individual feature vector `$v_{BE}$` acts as a **Query** seeking context, while the group relational context `$F_{CTX}$` acts as **Keys and Values** :
`$$v_{fused} = \operatorname{softmax}\left(\frac{(v_{BE}W_Q)(F_{CTX}W_K)^{\top}}{\sqrt{d}}\right) (F_{CTX}W_V)$$`
This step produces the final contextualized student embedding `$v_{fused}\in\mathbb{R}^d$`, incorporating individual expressions, full-body movements, skeletal configurations, nearby objects, and peer behaviors.

---

## Section 6: Ordinal-Aware Loss Formulations

Classroom engagement levels are _ordinal_ (Disengaged `$\to$` Normal `$\to$` Engaged `$\to$` Very Engaged). Predicting _Disengaged_ for a _Very Engaged_ student is a much larger error than predicting _Engaged_. CORE-Net implements a **Multi-Objective Hybrid Loss** function to enforce this ordinal relationship :

`$$\mathcal{L}_{HYB} = \lambda \mathcal{L}_{CE} + (1 - \lambda) \mathcal{L}_{ORD} + \alpha \mathcal{L}_{Nb}$$`

Let's break down each component mathematical term:

### 1. Cross-Entropy Loss (`$\mathcal{L}_{CE}$`)

`$$\mathcal{L}_{CE} = -\log p_y$$`
Where `$p_y$` is the predicted probability of the true ground-truth class `$y$`. This ensures sharp, well-separated categorical boundaries.

### 2. Ordinal Distance Loss (`$\mathcal{L}_{ORD}$`)

To penalize predictions proportionally to their distance from the true target, CORE-Net calculates the squared **Earth Mover's Distance ($EMD^2$)** over cumulative class probabilities :
`$$\mathcal{L}_{ORD} = \frac{1}{K} \sum_{k=1}^{K} (F_k - G_k)^2$$`
Where `$F_k = \sum_{i=1}^k p_i$` represents the predicted cumulative probability distribution, and `$G_k = 1\{y \le k\}$` is the cumulative ground-truth distribution. This term penalizes long-range deviations more severely than short-range errors.

### 3. Neighbor Regularizer Loss (`$\mathcal{L}_{Nb}$`)

To discourage the model from hedging between the true label `$y$` and its immediate neighbors `$y-1$` or `$y+1$`, this term penalizes probability leakage into adjacent classes :
`$$\mathcal{L}_{Nb} = p(y-1) + p(y+1)$$`

### 4. Dynamic Curriculum Scheduling

The hyperparameter `$\lambda$` balances classification versus ordinal supervision. CORE-Net uses a **cosine-decay schedule** to adjust this balance over epochs `$t$` :
`$$\lambda(t) = \lambda_{min} + (\lambda_{max} - \lambda_{min}) \left( \frac{1 + \cos(\pi t / T)}{2} \right)$$`
Early in training (`$t \approx 0$`), `$\lambda \approx \lambda_{max}$`, prioritizing Cross-Entropy so the model quickly learns stable class features. As training approaches the final epoch `$T$`, `$\lambda$` decays toward `$\lambda_{min}$`, shifting focus to `$\mathcal{L}_{ORD}$` to refine rank alignment.

---

## Section 7: Implementation Details & Benchmark Metrics

To reproduce the study, the authors configured the pipeline with the following execution settings :

- **Epochs**: 50 epochs.

- **Batch Size**: 4 samples per batch.

- **Optimizer**: AdamW.

- **Base Learning Rate**: `$1 \times 10^{-4}$` (`$1e-4$`).

- **Weight Decay**: `$1 \times 10^{-5}$` (`$1e-5$`).

- **Hardware Target**: NVIDIA A100 GPUs (40GB VRAM).

- **Dataset Partitioning**: Subject-Independent Group-Based Split.

### Performance Evaluation Metrics

Models are evaluated across three robust metrics:

1. **Accuracy (Acc.)**: Measures exact class matches.

2. **Mean Absolute Error (MAE)**: Measures average distance between predicted class indices and true indices, reflecting the severity of misclassifications.

3. **Quadratic Weighted Kappa (QWK)**: A standard behavioral analysis metric that measures the agreement between model predictions and human annotators.

### Benchmark Comparisons on COLER

CORE-Net significantly outperforms standard unimodal baselines on the COLER dataset :

| Model Class   | Architecture | Accuracy (%) | F1-Score (%) | QWK | MAE |
| ------------- | ------------ | ------------ | ------------ | --- | --- |
| **Pose-Only** | BlockGCN     |

| $74.09$ | $73.63$ | — | — |
| **Behavior-Only** | MoViNets

| $85.68$ | $85.08$ | — | — |
| **Facial-Only** | MoViNets

| $84.16$ | $83.53$ | — | — |
| **Multimodal** | **CORE-Net (Ours)**

| **89.63** | **89.34** | **94.80** | **17.61** |
