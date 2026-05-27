# TAAC PCVRHyFormer Competition Optimization Solution

[English](README_en.md) | [中文](README.md)

This is the optimization solution submitted for the TAAC PCVR (Post-Click Conversion Rate) task. Building upon the official baseline (PCVRHyFormer), this solution focuses on temporal feature engineering, multimodal semantic fusion, and engineering acceleration to construct a highly efficient and performant recommendation model architecture.

---

## 🚀 Core Optimization Ideas

[English](README_en.md) | [中文](README.md)

Our evolution is primarily based on the following four core dimensions of optimization, significantly enhancing the model's perception of temporal manifolds and representation efficiency of local features:

### 1. Request-Level Temporal Context Encoding (Context Time Tokens)

[English](README_en.md) | [中文](README.md)
**Concept**: The original baseline model suffered from the "loss of absolute time information." It only utilized timestamps to calculate time differences of past behaviors, resulting in the model's inability to explicitly perceive strong business features such as "morning/evening peaks" or "weekends." This solution parses and maps the absolute request timestamp into multi-dimensional discrete features (including hour, dow, dom, weekend), and feeds them into the model as independent Non-Sequence (NS) Tokens. This allows the network to capture strong intra-day/intra-week periodic patterns at shallow layers, effectively mitigating internal timing misalignment for users.

### 2. Int-Dense Dual-Channel Semantic Pre-alignment (Semantic Feature Alignment)

[English](README_en.md) | [中文](README.md)
**Concept**: In the baseline design, the discrete bins (Int) and continuous values within those bins (Dense) derived from the same signal enter independent channels and are only concatenated at the very end. This long chain naturally causes a magnitude collapse between the two, making continuous signals easily overshadowed by discrete ones. Before entering the global attention mechanism, this solution explicitly aligns them based on specific feature categories. Using a gated mapping network, pair-wise normalization and fusion are applied to the paired Int and Dense features. This early modal semantic alignment substantially improves the model's capacity to capture fine-grained residuals in continuous features.

### 3. Social Calendar Sparse Feature Mapping (Social Calendar)

[English](README_en.md) | [中文](README.md)
**Concept**: Periodic temporal features fail to represent irregular events like statutory holidays or extended promotional campaigns. For example, a makeup workday occurring on a weekend is intrinsically a workday, while a multi-week promotional season completely reshapes consumption distributions. This solution independently introduces a small public calendar vocabulary system, encoding and embedding special social states (including statutory holidays, makeup workdays, and platform-level promotional seasons). This not only offloads the burden from periodic temporal features but also immensely improves the model's robustness against data distribution shifts during significant operational milestones.

### 4. System-Level Engineering and Training Acceleration (Training Acceleration)

[English](README_en.md) | [中文](README.md)
**Concept**: To achieve rapid iterations for feature and structure experiments, we implemented non-intrusive acceleration modifications tailored to underlying hardware characteristics. This set of optimization seamlessly integrates BF16 mixed-precision training, TF32 calculation acceleration, PyTorch's built-in graph compilation (	orch.compile), and dynamic scaling for large Batch Sizes. With all acceleration components decoupled via command-line arguments, we managed to reduce the wall-clock batch running time by approximately 35% ~ 50% without compromising any evaluation precision (AUC metrics and model comparability).

---

## 📦 How to Run

[English](README_en.md) | [中文](README.md)

You can start the training with all optimizations enabled by default as specified in the startup script:

`ash
bash run.sh
`

For engineering acceleration, environmental optimization arguments can be appended directly to the command:

`ash
# e.g., Enable BF16 mixed precision, graph compilation optimization, and increase Batch Size to boost throughput

[English](README_en.md) | [中文](README.md)
bash run.sh --amp bf16 --torch_compile --batch_size 512
`

---

## 📑 Platform Submission Guidelines

[English](README_en.md) | [中文](README.md)

When packaging for the TAAC evaluation environment, all code must be placed in a flat directory.
Ensure the uploaded package contains the following core dependencies required for running logic and feature parsing:
	rain.py, 	rainer.py, model.py, dataset.py, utils.py, lignment.py, lignment_pairs.json, un.sh, cn_request_calendar.csv, along with infer.py used for evaluation.