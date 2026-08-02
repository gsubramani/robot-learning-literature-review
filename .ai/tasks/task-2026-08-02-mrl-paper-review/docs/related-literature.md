# Related Literature: Representation Learning & Model Compression

Context for Matryoshka Representation Learning (MRL) — [arxiv:2205.13147](https://arxiv.org/abs/2205.13147)

---

## 1. Representation Learning

### 1.1 Supervised Representation Learning

Learning general-purpose features by training a classifier on large labeled datasets, then reusing the penultimate layer embeddings for downstream tasks.

- **AlexNet / VGG / ResNet** [Krizhevsky 2012, Simonyan 2014, He 2016] — ImageNet classification as the canonical pretraining task. ResNet50 produces 2048-dim embeddings widely used as frozen features.
- **Vision Transformers (ViT)** [Dosovitskiy 2020] — Patch-based transformer for image classification; [CLS] token embedding serves as the representation.
- **JFT-300M pretraining** [Sun 2017, Kolesnikov 2020] — Web-scale supervised pretraining (300M images) produces representations that transfer better than ImageNet-only models.

**MRL connection**: MRL plugs directly into supervised pretraining by replacing the single linear classification head with multi-scale heads over nested embedding dimensions.

### 1.2 Self-Supervised / Unsupervised Representation Learning

Learning representations without labels through proxy tasks.

- **Contrastive learning (SimCLR, MoCo, CLIP)** [Chen 2020, He 2020, Radford 2021] — Pull positive pairs together, push negatives apart in embedding space. CLIP learns cross-modal (image-text) representations at web scale.
- **Masked language modeling (BERT, RoBERTa)** [Devlin 2018, Liu 2019] — Predict masked tokens from context. BERT-base produces 768-dim token embeddings.
- **Autoregressive language models (GPT)** [Radford 2018, Brown 2020] — Predict next token; representations emerge in hidden states.
- **Masked image modeling (MAE)** [He 2021] — Reconstruct masked image patches; learns strong visual representations without labels.
- **Non-contrastive methods (BYOL, DINO, SimSiam)** [Grill 2020, Caron 2021, Chen 2021] — Learn representations without negative pairs, using stop-gradients and teacher-student architectures.

**MRL connection**: The paper applies MRL to contrastive learning (ALIGN) and MLM (BERT). For contrastive losses, MRL is applied to both embeddings being contrasted, with per-dimension normalization. For MLM, weight tying between input embeddings and output head naturally yields MRL-E.

### 1.3 Representation Properties & Analysis

- **Intrinsic dimensionality** [Li 2018, Ansuini 2019] — Studies the effective dimensionality of learned representations; finds that deep networks often use far fewer dimensions than the nominal embedding size, suggesting redundancy.
- **Catastrophic forgetting / continual learning** [Kirkpatrick 2017] — Representation quality degrades when learning new tasks; MRL shows 2% gains on long-tail novel classes.
- **Linear probe evaluation** [Kornblith 2019] — Standard protocol for evaluating representation quality: freeze backbone, train linear classifier on top.
- **Information bottleneck theory** [Tishby 2015, Saxe 2018] — Representations should compress input while preserving task-relevant information; MRL provides an empirical framework for studying this via coarse-to-fine packing.

### 1.4 Ordered / Nested Representations

Most directly related to MRL:

- **Nested dropout** [Rippel et al. 2015] — Applies dropout to ordered dimensions of autoencoder latent codes, forcing earlier dimensions to carry more information. Optimizes O(d) nested dimensions (one per dimension), making it expensive at scale. MRL optimizes only O(log d) and interpolates for the rest.
- **Slimmable networks** [Yu et al. 2018] — Train multiple sub-networks of varying width within a single network using switchable batch normalization. Requires separate forward passes per capacity level and doesn't produce a single multi-granular embedding.
- **Progressive neural networks / sub-net packing** [Rusu 2016, Yu 2019] — Pack networks of varying capacity into a larger one, but each sub-network has distinct weights and requires its own forward pass.

**MRL differentiation**: MRL produces a *single* embedding vector where slicing gives you different granularities — no re-encoding, no multiple forward passes, no separate weight sets for the backbone.

---

## 2. Model Compression

MRL compresses representations, not weights. But it sits in a broader landscape of compression techniques. Below are the major categories, all of which are **complementary** to MRL.

### 2.1 Pruning

Removing redundant weights or neurons from a trained model.

- **Magnitude pruning** [Han et al. 2015] — Remove weights below a magnitude threshold; fine-tune to recover accuracy. Can achieve 9x–13x compression on CNNs.
- **Lottery ticket hypothesis** [Frankle & Carbin 2018] — Dense networks contain sparse subnetworks ("winning tickets") that match full accuracy when trained in isolation.
- **Structured pruning (channel/filter pruning)** [Li et al. 2016, Liu et al. 2017] — Remove entire filters or channels, enabling actual speedups on standard hardware (unlike unstructured pruning which needs sparse kernels).
- **Movement pruning** [Sanh et al. 2020] — Differentiable pruning for BERT; removes weights that move toward zero during fine-tuning.
- **SparseGPT / Wanda** [Frantar & Alistarh 2023, Sun et al. 2023] — Post-hoc LLM pruning without retraining; achieve 50% sparsity on LLaMA with minimal accuracy loss.

**vs. MRL**: Pruning reduces forward-pass FLOPs. MRL reduces downstream embedding usage cost. They can be stacked: prune the backbone, then apply MRL to the output.

### 2.2 Quantization

Reducing the numerical precision of weights and/or activations.

- **Post-training quantization (PTQ)** [Jacob et al. 2018] — Quantize a trained model to INT8 without retraining. Minimal accuracy loss for most CNNs.
- **Quantization-aware training (QAT)** [Krishnamoorthi 2018] — Simulate quantization during training so the model learns to be robust to low precision.
- **Binary / ternary networks** [Courbariaux 2016, Zhu 2016] — Weights constrained to {-1, +1} or {-1, 0, +1}; massive memory savings but significant accuracy drops.
- **LLM.int8() / GPTQ / AWQ** [Dettmers 2022, Frantar 2022, Lin 2023] — Quantization methods specifically for LLMs; handle outlier features that make LLM quantization harder.
- **SmoothQuant** [Xiao 2022] — Migrate difficulty from activations to weights via per-channel scaling, enabling INT8 quantization of LLMs.

**vs. MRL**: Quantization reduces memory and matmul cost. MRL reduces embedding dimensionality. Fully orthogonal — you can quantize an MRL-trained model.

### 2.3 Knowledge Distillation

Training a smaller "student" model to mimic a larger "teacher."

- **Hinton distillation** [Hinton et al. 2015] — Student matches teacher's soft logits (with temperature scaling) plus ground-truth labels.
- **Feature-level distillation** [Romero et al. 2014, FitNets] — Student matches intermediate feature maps, not just outputs.
- **DistilBERT** [Sanh et al. 2019] — 40% smaller BERT with 97% of capability, via distillation on soft labels.
- **TinyBERT / MobileBERT** [Jiao et al. 2019, Sun et al. 2020] — Layer-level distillation for BERT; student has fewer layers and smaller hidden dim.
- **Self-distillation / Born-again networks** [Furlanello et al. 2018] — Student has same architecture as teacher; iteratively improves via self-teaching.

**vs. MRL**: Distillation changes the model itself (fewer params/layers). MRL keeps the model identical and only changes the training loss. A distilled student could itself be trained with MRL.

### 2.4 Low-Rank Approximation

Approximating weight matrices with low-rank factorizations.

- **SVD-based compression** [Denil et al. 2013, Tai et al. 2015] — Decompose weight matrices W ≈ U·V^T where U ∈ R^{m×r}, V ∈ R^{n×r}, r << min(m,n). Reduces both storage and FLOPs.
- **Low-rank adaptation (LoRA)** [Hu et al. 2021] — Freeze pretrained weights, inject trainable low-rank matrices. Originally for efficient fine-tuning, but the low-rank structure itself is a form of compression.
- **Tensor decomposition (Tucker, CP)** [Lebedev et al. 2014] — Generalize SVD to conv layers via tensor factorization; can achieve 10x+ compression on CNNs.
- **Post-hoc SVD on embeddings** — The paper compares MRL against SVD compression of embeddings and shows MRL is significantly more accurate at low dimensions, because SVD doesn't retrain the model to pack information into the retained dimensions.

**vs. MRL**: SVD on embeddings is the closest non-MRL baseline. The key difference: SVD finds the best *linear* approximation of existing embeddings post-hoc, while MRL *trains* the model to put useful information in early dimensions, which is a non-linear optimization. MRL wins by 2–3% mAP at ≤256 dims.

### 2.5 Dimensionality Reduction & Hashing

Post-hoc techniques to reduce embedding dimensionality for retrieval.

- **PCA / random projection** [Jolliffe 2002, Johnson-Lindenstrauss 1984] — Linear dimensionality reduction. J-L lemma guarantees distance preservation for random projections, but with O(ε⁻²) dimensions.
- **Product quantization (PQ)** [Jégou et al. 2011] — Split embedding into sub-vectors, quantize each independently. Widely used in FAISS for billion-scale retrieval.
- **Locality-sensitive hashing (LSH)** [Indyk & Motwani 1998] — Hash similar items to same bucket; enables sub-linear search but with recall/precision trade-offs.
- **Binary hashing (ITQ, SH, DSH)** [Gong 2012, Weiss 2008, Jin 2013] — Map real-valued embeddings to binary codes for Hamming-distance retrieval; 32x memory savings vs. float32.

**vs. MRL**: These are all *post-hoc* — they compress already-trained embeddings without retraining. MRL is *trained-in*, producing more accurate low-dim representations. All these techniques are complementary to MRL: apply PQ or binary hashing on top of MRL's low-dim slices for further compression.

### 2.6 Efficient Architecture Design

Designing architectures that are inherently more efficient.

- **MobileNet / EfficientNet** [Howard 2017, Tan 2019] — Depthwise separable convolutions and compound scaling for mobile CNNs.
- **Distill-and-prune combined** — Many production pipelines combine distillation + pruning + quantization.
- **Mixture of Experts (MoE)** [Shazeer 2017, Fedus 2022] — Sparse activation: only a subset of expert sub-networks process each input, reducing average FLOPs while increasing total parameter count.
- **Early exit / dynamic depth** [Xin 2020, Zhou 2020] — Transformers with intermediate classifiers; exit early for easy inputs. Conceptually similar to MRL's adaptive classification cascade, but operates on *layers* rather than *embedding dimensions*.

**vs. MRL**: Early exit reduces forward-pass cost by skipping layers. MRL reduces downstream cost by using fewer embedding dimensions. They could be combined: early-exit for compute savings + MRL for embedding savings.

### 2.7 Neural Architecture Search (NAS) for Compression

- **Once-for-all networks** [Cai et al. 2019] — Train a single network that can be specialized to multiple architectures (depth, width, kernel size) at deployment. Requires specialized training with progressive shrinking.
- **NetAdapt** [Yang et al. 2018] — Automatically prune channels to meet latency constraints on target hardware.

**vs. MRL**: NAS-based methods change the model architecture. MRL keeps the architecture fixed and only modifies the training loss. NAS + MRL could yield architectures optimized for multi-granular embeddings.

---

## 3. Positioning of MRL in the Landscape

```
┌─────────────────────────────────────────────────────────────┐
│                  COMPRESSION TARGET                         │
│                                                             │
│   Model Weights          Output Embeddings                  │
│   (forward-pass cost)    (downstream cost)                  │
│                                                             │
│  ┌──────────────┐        ┌──────────────────────┐          │
│  │ Pruning      │        │ MRL (trained-in)     │ ◄── MRL  │
│  │ Quantization │        │ SVD (post-hoc)       │          │
│  │ Distillation │        │ PCA / random proj.   │          │
│  │ Low-rank     │        │ Product quantization │          │
│  │ NAS / MoE    │        │ Binary hashing       │          │
│  │ Early exit   │        │ LSH                  │          │
│  └──────────────┘        └──────────────────────┘          │
│                                                             │
│  All weight-side and embedding-side techniques are          │
│  COMPLEMENTARY — they can be stacked.                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight

MRL uniquely occupies the **trained-in embedding compression** niche:
- Unlike SVD/PCA/PQ, it optimizes information packing *during training*, not post-hoc.
- Unlike pruning/quantization/distillation, it doesn't change the model at all.
- Unlike slimmable networks, it produces a single embedding — no re-encoding for different granularities.
- Unlike nested dropout, it only optimizes O(log d) dimensions instead of O(d), making web-scale training feasible.

---

## 4. Key References

### Representation Learning
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition. ICLR.
- Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
- Chen et al. (2020). SimCLR: A Simple Framework for Contrastive Learning. ICML.
- Radford et al. (2021). CLIP: Learning Transferable Visual Models From Natural Language Supervision. ICML.
- Grill et al. (2020). BYOL: Bootstrap Your Own Latent. NeurIPS.
- Caron et al. (2021). DINO: Emerging Properties in Self-Supervised Vision Transformers. ICCV.

### Ordered/Nested Representations (most related to MRL)
- Rippel et al. (2015). Learning Ordered Representations with Nested Dropout. ICML.
- Yu et al. (2018). Slimmable Neural Networks. ICLR.
- Yu et al. (2019). Universally Slimmable Networks and Improved Training Techniques. ICCV.

### Pruning
- Han et al. (2015). Learning both Weights and Connections for Efficient Neural Networks. NeurIPS.
- Frankle & Carbin (2018). The Lottery Ticket Hypothesis. ICLR.
- Sanh et al. (2020). Movement Pruning: Adaptive Sparsity by Fine-Tuning. NeurIPS.
- Frantar & Alistarh (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. ICML.

### Quantization
- Jacob et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR.
- Dettmers et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS.
- Frantar et al. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR.
- Lin et al. (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys.

### Knowledge Distillation
- Hinton et al. (2015). Distilling the Knowledge in a Neural Network. NeurIPS Workshop.
- Sanh et al. (2019). DistilBERT: A distilled version of BERT. NeurIPS Workshop.
- Jiao et al. (2019). TinyBERT: Distilling BERT for Natural Language Understanding. Findings of ACL.

### Low-Rank & Dimensionality Reduction
- Denil et al. (2013). Predicting Parameters in Deep Learning. NeurIPS.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Jégou et al. (2011). Product Quantization for Nearest Neighbor Search. IEEE TPAMI.
- Johnson et al. (1984). Extensions of Lipschitz Mapping into a Hilbert Space. ICASSP.

### Efficient Inference
- Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer. ICLR.
- Xin et al. (2020). DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference. ACL.
- Cai et al. (2019). Once-for-All: Train One Network and Specialize it for Efficient Deployment. ICLR.
