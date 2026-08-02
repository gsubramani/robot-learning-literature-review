# Matryoshka Representation Learning (MRL) — Paper Analysis

**Paper**: [arxiv:2205.13147](https://arxiv.org/abs/2205.13147)  
**Authors**: Kusupati et al.  
**Code**: https://github.com/RAIVNLab/MRL

---

## 1. What MRL Compresses (and What It Doesn't)

MRL compresses **representations (embeddings)**, not model weights. The neural network's parameters stay the same size. What changes is that the *output embedding vector* becomes multi-granular: the first `m` dimensions are a useful representation on their own, for multiple values of `m`.

This means:
- **No savings in forward-pass FLOPs** — you still run the full model.
- **Savings in downstream compute** — classification, retrieval, and storage all scale with embedding dimensionality, not model size.
- **Up to 14x smaller embedding size** at the same accuracy for ImageNet-1K classification.
- **Up to 14x real-world speedups** for large-scale retrieval.

---

## 2. How MRL Works

### 2.1 Core Mechanism

Given a model `F(x; θ)` that produces a `d`-dimensional embedding `z = F(x)`, MRL chooses a set of nesting dimensions:

```
M = {8, 16, 32, 64, ..., 1024, 2048}   (|M| ≤ log₂(d))
```

For each `m ∈ M`, the first `m` dimensions `z[1:m]` must independently serve as a transferable representation. This is enforced via a **multi-scale loss**:

```
L_MRL = Σ_{m ∈ M}  c_m · L(W^(m) · z[1:m], y)
```

where:
- `W^(m) ∈ R^{L×m}` is a separate linear classifier for dimension `m`
- `L` is standard cross-entropy loss
- `c_m` are importance weights (typically all set to 1)

The model is trained with this combined loss, forcing it to pack information **coarse-to-fine**: the first 8 dims capture coarse semantics, dims 9–16 add finer detail, and so on.

### 2.2 Efficient Variant (MRL-E)

Instead of separate classifiers per dimension, use **weight tying**:

```
W^(m) = W[:, 1:m]    (slice of a single W ∈ R^{L×d})
```

This halves the classifier memory and is the natural choice when the model already uses weight tying (e.g., BERT's MLM head ties input embeddings to the output classifier).

### 2.3 Training Code (from paper Appendix A)

```python
class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance, **kwargs):
        super(Matryoshka_CE_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        self.relative_importance = relative_importance  # usually all ones

    def forward(self, output, target):
        loss = 0
        for i in range(len(output)):
            loss += self.relative_importance[i] * self.criterion(output[i], target)
        return loss


class MRL_Linear_Layer(nn.Module):
    def __init__(self, nesting_list, num_classes=1000, efficient=False, **kwargs):
        super(MRL_Linear_Layer, self).__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        self.is_efficient = efficient

        if not self.is_efficient:
            for i, num_feat in enumerate(self.nesting_list):
                setattr(self, f"nesting_classifier_{i}",
                        nn.Linear(num_feat, self.num_classes, **kwargs))
        else:
            setattr(self, "nesting_classifier_0",
                    nn.Linear(self.nesting_list[-1], self.num_classes, **kwargs))

    def forward(self, x):
        nesting_logits = ()
        for i, num_feat in enumerate(self.nesting_list):
            if self.is_efficient:
                efficient_logit = torch.matmul(
                    x[:, :num_feat],
                    (self.nesting_classifier_0.weight[:, :num_feat]).t()
                )
                nesting_logits += (efficient_logit,)
            else:
                nesting_logits += (
                    getattr(self, f"nesting_classifier_{i}")(x[:, :num_feat]),
                )
        return nesting_logits
```

### 2.4 Key Properties

- **Minimal training overhead**: Only O(log d) extra linear classifiers (or zero extra with MRL-E).
- **No inference overhead**: The forward pass is unchanged; you just slice the output.
- **Interpolation**: Even dimensions not explicitly trained (e.g., 24, 48) work well — information interpolates across all d dimensions.
- **Robustness**: MRL representations are at least as robust as standard ones, with up to 2% improvement on out-of-distribution benchmarks.

---

## 3. Adaptive Deployment

### 3.1 Adaptive Classification (MRL-AC)

Use a cascade: classify with 8-dim → if confidence (max softmax prob) is below threshold → upgrade to 16-dim → 32-dim → ... → 2048-dim.

Thresholds are learned on a validation set. Result: **~14x smaller average embedding size** at the same accuracy (e.g., expected dim ~37 vs 512 for 76.3% on ImageNet-1K).

### 3.2 Adaptive Retrieval (MRL-AR)

1. Shortlist candidates using low-dim embeddings (fast approximate NN search).
2. Re-rank the shortlist using progressively higher dimensions.

Result: **128x theoretical (FLOPS) and 14x wall-clock speedups** for retrieval at comparable accuracy.

---

## 4. Applying MRL to an LSTM Layer

### 4.1 What You Compress

MRL compresses the **output hidden state** of the LSTM, not the LSTM weights. The LSTM still has the same number of parameters.

### 4.2 Setup

Suppose your LSTM produces a hidden state `h ∈ R^d` (e.g., d=512). Choose nesting dimensions:

```python
M = [8, 16, 32, 64, 128, 256, 512]
```

### 4.3 Training Modification

Replace the standard classification head with the MRL linear layer:

```python
import torch
import torch.nn as nn

class MRL_LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, nesting_list):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.nesting_list = nesting_list
        self.num_classes = num_classes

        # MRL-E: single weight matrix, sliced for each nesting dim
        self.classifier = nn.Linear(nesting_list[-1], num_classes, bias=False)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch, hidden_dim)
        h = h_n.squeeze(0)          # (batch, hidden_dim)

        nesting_logits = []
        for m in self.nesting_list:
            # Slice first m dims of hidden state and classifier weights
            logit = torch.matmul(h[:, :m], self.classifier.weight[:, :m].t())
            nesting_logits.append(logit)

        return nesting_logits  # list of (batch, num_classes) at each granularity

    def compute_loss(self, logits_list, targets, weights=None):
        if weights is None:
            weights = [1.0] * len(logits_list)
        loss = 0
        for w, logits in zip(weights, logits_list):
            loss += w * nn.functional.cross_entropy(logits, targets)
        return loss
```

### 4.4 Inference

At inference, choose the granularity based on your compute budget:

```python
# Full capacity (512-dim)
logits = model(x)[-1]  # last element = full dim

# Compressed (64-dim) — just use the 6th element
logits = model(x)[3]   # 64-dim classifier output

# Or: extract the embedding and slice it
h = model.lstm(x)[1][0].squeeze(0)
h_compressed = h[:, :64]  # use first 64 dims for downstream tasks
```

### 4.5 What This Achieves for LSTM

- **Embedding storage**: Store 64-dim instead of 512-dim vectors (8x smaller).
- **Downstream classification**: Use a 64×L instead of 512×L classifier matrix.
- **Retrieval**: Search with 64-dim vectors instead of 512-dim (8x faster distance computation).
- **No change to LSTM forward pass cost** — the LSTM itself is still the same size.

### 4.6 Limitation for LSTM

LSTM hidden states are recurrently updated, so the information packing is constrained by the recurrent dynamics. The model must learn to put coarse info in early dimensions *through the gating mechanism*, which may be harder than with feed-forward models. The paper doesn't test LSTM specifically, but the principle is architecture-agnostic — it only requires a differentiable output vector.

---

## 5. Applying MRL to a Transformer Layer

### 5.1 What You Compress

The output representation of the transformer (e.g., `[CLS]` token embedding, pooled output, or last hidden state). The paper already demonstrates this with **BERT**.

### 5.2 Setup

For a transformer with hidden dim `d=768` (BERT-base):

```python
M = [8, 16, 32, 64, 128, 256, 512, 768]
```

### 5.3 Training Modification

For **masked language modeling (MLM)**, BERT already ties the input embedding matrix to the output classifier. This makes MRL-E the natural variant:

```python
class MRL_TransformerMLM(nn.Module):
    def __init__(self, transformer, nesting_list, vocab_size):
        super().__init__()
        self.transformer = transformer  # e.g., HuggingFace BERT
        self.nesting_list = nesting_list
        self.vocab_size = vocab_size

        # Single weight matrix (tied with input embeddings in practice)
        self.decoder = nn.Linear(nesting_list[-1], vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        nesting_logits = []
        for m in self.nesting_list:
            # Slice hidden states to first m dims
            h_slice = hidden_states[..., :m]
            # Slice decoder weights to first m dims
            w_slice = self.decoder.weight[:, :m]
            logits = torch.matmul(h_slice, w_slice.t())
            nesting_logits.append(logits)

        return nesting_logits

    def compute_loss(self, logits_list, labels, weights=None):
        if weights is None:
            weights = [1.0] * len(logits_list)
        loss = 0
        for w, logits in zip(weights, logits_list):
            # Standard MLM loss at each granularity
            loss += w * nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        return loss
```

### 5.4 For Classification (e.g., fine-tuning BERT)

```python
class MRL_BERTClassifier(nn.Module):
    def __init__(self, bert_model, nesting_list, num_classes):
        super().__init__()
        self.bert = bert_model
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        self.classifier = nn.Linear(nesting_list[-1], num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token

        nesting_logits = []
        for m in self.nesting_list:
            h_slice = cls_embedding[:, :m]
            w_slice = self.classifier.weight[:, :m]
            b = self.classifier.bias
            logits = torch.matmul(h_slice, w_slice.t()) + b
            nesting_logits.append(logits)

        return nesting_logits
```

### 5.5 Inference — Adaptive Classification

```python
# Cascade: try 32-dim first, upgrade if uncertain
logits_list = model(input_ids, attention_mask)

thresholds = [0.9, 0.85, 0.8, 0.75]  # learned on validation set
for i, (logits, thresh) in enumerate(zip(logits_list[:-1], thresholds)):
    probs = torch.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1).values
    if max_prob >= thresh:
        return logits  # confident enough at this granularity

return logits_list[-1]  # fall back to full-dim
```

### 5.6 What This Achieves for Transformer

- **Embedding storage**: Store 64-dim or 128-dim sentence embeddings instead of 768-dim (6-12x smaller).
- **Retrieval**: Search with low-dim vectors for initial candidate shortlisting, re-rank with full-dim.
- **Adaptive classification**: Easy samples classified with 8-dim, hard ones with 768-dim.
- **The paper shows BERT-MRL works** — the MLM weight tying naturally gives you MRL-E.

### 5.7 Additional Transformer Considerations

- **Layer-wise MRL**: You could apply MRL across transformer *layers* (not just within the embedding dim). Early layers = coarse, later layers = fine. This is analogous but not what the paper does.
- **Normalization**: The paper notes that if the representation is normalized (e.g., for contrastive learning), you must normalize each nesting dimension independently: `z[1:m] / ||z[1:m]||`.
- **Multi-head attention**: MRL doesn't directly compress attention heads, but you could use the first m dims of the output to implicitly prioritize certain heads during training.

---

## 6. MRL vs. Other Compression Techniques

| Technique | What's Compressed | Training Overhead | Inference Cost Change |
|-----------|------------------|-------------------|----------------------|
| **MRL** | Output embeddings | O(log d) extra classifiers | None (just slice output) |
| **Pruning** | Model weights | Retraining needed | Lower FLOPs |
| **Quantization** | Weight precision | Calibration/fine-tuning | Lower memory, faster matmul |
| **Distillation** | Model size | Train student from scratch | Lower FLOPs |
| **SVD/post-hoc** | Output embeddings | None | None, but accuracy drops much more |

MRL is **complementary** to pruning, quantization, and distillation — you can apply MRL on top of a pruned/quantized model.

---

## 7. Key Takeaways

1. **MRL is representation compression, not weight compression.** The model stays the same size; the embeddings become multi-granular.
2. **Training change is minimal**: add O(log d) linear heads and sum their losses. With MRL-E, even the extra heads disappear (weight tying).
3. **For LSTM**: Apply MRL to the final hidden state. The model learns to pack coarse info in early dims through the gating mechanism.
4. **For Transformer**: Apply MRL to the [CLS] token or MLM output. BERT's weight tying makes MRL-E natural. The paper already validates this.
5. **Main benefit**: Adaptive deployment — use small embeddings for easy tasks/large-scale retrieval, full embeddings for hard cases.
6. **Limitation**: No savings in forward-pass compute. If your bottleneck is model inference (not downstream embedding usage), MRL won't help — use pruning/quantization/distillation instead.
