# MRL for Intermediate Layer Dimension Reduction

## The Problem

Standard MRL compresses the **final output embedding** — the last representation before a classifier head. But what if you want to compress the output of a **single intermediate layer** in a larger network?

The challenge: an intermediate layer's output feeds into **downstream layers** that expect the full `d`-dimensional input. You can't just slice the embedding and hand it to the next layer without adaptation.

```
Input → [Layer A] → h ∈ R^d → [Layer B] → [Layer C] → output
                ^
                |
         compress this to m < d
```

---

## Approach 1: Weight-Slicing (MRL-E Extended to Intermediate Layers)

**Best when the next layer is linear.** This is the most faithful extension of MRL.

The idea: the next layer's weight matrix `W ∈ R^{out × d}` can be sliced to `W[:, :m]` to accept `m`-dimensional input. During training, you compute the task loss at each granularity by slicing both the intermediate output and the downstream weights.

```python
import torch
import torch.nn as nn

class MRL_IntermediateLayer(nn.Module):
    """
    Wraps a layer and its downstream linear consumer to enable MRL-style
    dimension reduction on the intermediate representation.
    """
    def __init__(self, producer: nn.Module, consumer: nn.Module,
                 nesting_list: list[int]):
        super().__init__()
        self.producer = producer        # e.g., nn.Linear(in, d) or conv block
        self.consumer = consumer        # e.g., nn.Linear(d, out) — must be linear
        self.nesting_list = nesting_list

        # Verify consumer is linear so we can slice its weights
        assert isinstance(consumer, nn.Linear), \
            "Weight-slicing approach requires the consumer to be nn.Linear"

    def forward(self, x, return_all_granularities=False):
        h = self.producer(x)  # (batch, d)

        if not return_all_granularities:
            # Standard forward: full dimension
            return self.consumer(h)

        # MRL forward: compute output at each nesting dimension
        outputs = []
        for m in self.nesting_list:
            h_slice = h[..., :m]                    # (batch, m)
            w_slice = self.consumer.weight[:, :m]   # (out, m)
            out = torch.matmul(h_slice, w_slice.t())
            if self.consumer.bias is not None:
                out = out + self.consumer.bias
            outputs.append(out)
        return outputs

    def compute_mrl_loss(self, outputs, task_loss_fn, *targets):
        """Sum task loss across all granularities."""
        total = 0
        for out in outputs:
            total += task_loss_fn(out, *targets)
        return total
```

### Usage in a larger network

```python
class MyNetwork(nn.Module):
    def __init__(self, dim=512, num_classes=100):
        super().__init__()
        self.layer_a = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, dim), nn.ReLU(),
        )
        # This is the layer whose output we want to compress
        self.mrl_block = MRL_IntermediateLayer(
            producer=nn.Identity(),       # layer_a already produces the rep
            consumer=nn.Linear(dim, 128), # next layer
            nesting_list=[32, 64, 128, 256, 512],
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x, training=False):
        h = self.layer_a(x)  # (batch, 512)

        if training:
            # Get outputs at all granularities
            intermediate_outputs = self.mrl_block(h, return_all_granularities=True)
            # Each is (batch, 128) — the consumer maps m-dim → 128-dim
            # Then pass each through the head
            all_logits = [self.head(out) for out in intermediate_outputs]
            return all_logits
        else:
            # At inference, choose granularity
            out = self.mrl_block(h, return_all_granularities=False)
            return self.head(out)

    def compute_loss(self, all_logits, labels):
        return sum(nn.functional.cross_entropy(l, labels) for l in all_logits)
```

### Inference with reduced dimension

```python
# At inference, use only first 64 dims of the intermediate layer
h = model.layer_a(x)                    # (batch, 512)
h_compressed = h[..., :64]              # (batch, 64) — 8x smaller
# Slice the consumer weights to match
w = model.mrl_block.consumer.weight[:, :64]  # (128, 64)
out = torch.matmul(h_compressed, w.t()) + model.mrl_block.consumer.bias
logits = model.head(out)
```

**What this achieves**: The intermediate representation can be stored/transmitted at `m` dims instead of `d` dims. The downstream layer adapts by using only the first `m` columns of its weight matrix. The model learns to pack useful information into the early dimensions during training.

### Limitations

- Only works when the **immediate consumer is a linear layer** (so weights can be sliced).
- If there are multiple downstream consumers, each needs its own sliced weights.
- Non-linear activations between producer and consumer are fine (they're element-wise).

---

## Approach 2: Projection Adapter

**Best when the next layer is non-linear or has architectural constraints** (e.g., attention, conv layers that expect a specific channel count).

Instead of slicing the downstream weights, learn a small projection that maps `m`-dim → `d`-dim, restoring the expected input size for downstream layers.

```python
class MRL_ProjectionAdapter(nn.Module):
    """
    For each nesting dimension m, learn a projection from m-dim back to d-dim
    so downstream layers receive their expected input size.
    """
    def __init__(self, full_dim: int, nesting_list: list[int]):
        super().__init__()
        self.full_dim = full_dim
        self.nesting_list = nesting_list

        # One projection per nesting dimension (m → d)
        self.projections = nn.ModuleDict({
            str(m): nn.Linear(m, full_dim, bias=False)
            for m in nesting_list if m < full_dim
        })

    def forward(self, h, target_dim=None):
        """
        h: (batch, d) — full intermediate representation
        target_dim: which nesting dimension to use (None = full)
        """
        if target_dim is None or target_dim == self.full_dim:
            return h  # no compression

        h_slice = h[..., :target_dim]  # (batch, m)
        return self.projections[str(target_dim)](h_slice)  # (batch, d)
```

### Usage

```python
class NetworkWithAdapter(nn.Module):
    def __init__(self, dim=512, nesting_list=[32, 64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, dim), nn.ReLU()
        )
        self.adapter = MRL_ProjectionAdapter(dim, nesting_list)
        self.nesting_list = nesting_list
        self.decoder = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x, target_dim=None):
        h = self.encoder(x)  # (batch, 512)

        if target_dim is not None:
            # Use compressed intermediate rep
            h_adapted = self.adapter(h, target_dim)
            return self.decoder(h_adapted)

        # Training: compute loss at all granularities
        all_logits = []
        for m in self.nesting_list:
            h_adapted = self.adapter(h, m if m < self.nesting_list[-1] else None)
            all_logits.append(self.decoder(h_adapted))
        return all_logits
```

### Training loss

```python
def mrl_loss(all_logits, labels, weights=None):
    if weights is None:
        weights = [1.0 / len(all_logits)] * len(all_logits)
    return sum(w * nn.functional.cross_entropy(l, labels)
               for w, l in zip(weights, all_logits))
```

### Trade-offs

- **Pro**: Works with any downstream architecture (attention, conv, etc.)
- **Pro**: Downstream layers don't need modification
- **Con**: The projection adds parameters (m × d per nesting level)
- **Con**: At inference with `m`-dim, you still pay for the `m → d` projection + full downstream computation. The savings are in **storage/transmission of the intermediate representation**, not downstream FLOPs.

**When to use**: When the intermediate representation is a bottleneck for memory or communication (e.g., distributed inference, pipeline parallelism, storing activations for later use), but downstream compute is not the concern.

---

## Approach 3: Nested Dropout on Intermediate Layer

**Simplest approach. No extra parameters. Works with any architecture.**

Randomly sample a nesting dimension `m` during each forward pass, zero out dimensions `m+1:` of the intermediate output, and continue the forward pass. This forces the model to not depend on any specific dimension being present.

```python
class NestedDropoutLayer(nn.Module):
    def __init__(self, dim: int, nesting_list: list[int]):
        super().__init__()
        self.dim = dim
        self.nesting_list = sorted(nesting_list)

    def forward(self, h, training=True):
        if not training:
            return h  # full dim at inference

        # Sample a random nesting dimension
        m = self.nesting_list[torch.randint(0, len(self.nesting_list), (1,)).item()]

        # Zero out dimensions beyond m
        mask = torch.zeros_like(h)
        mask[..., :m] = 1.0
        return h * mask

    def forward_at_dim(self, h, m):
        """Use specific dimension at inference."""
        h_truncated = h.clone()
        h_truncated[..., m:] = 0
        return h_truncated
```

### Usage

```python
class NetworkWithNestedDropout(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, dim), nn.ReLU()
        )
        self.nested_dropout = NestedDropoutLayer(dim, [32, 64, 128, 256, 512])
        self.decoder = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x, target_dim=None):
        h = self.encoder(x)
        if target_dim is not None:
            h = self.nested_dropout.forward_at_dim(h, target_dim)
        else:
            h = self.nested_dropout(h, training=self.training)
        return self.decoder(h)
```

### Trade-offs

- **Pro**: Zero extra parameters, trivial to implement
- **Pro**: Works with any downstream architecture
- **Pro**: Downstream layers learn to handle partial inputs naturally
- **Con**: This is closer to Rippel et al.'s nested dropout than to MRL — it optimizes O(d) implicit nesting dimensions, not O(log d) explicit ones
- **Con**: Higher variance in training (random truncation), may need more epochs
- **Con**: Less precise control over which dimensions are "good" — no explicit loss per granularity

**When to use**: Quick prototyping, or when you don't want to modify the loss function. The model learns robustness to missing dimensions implicitly.

---

## Approach 4: Bottleneck with MRL (Recommended for New Architectures)

**Insert a dedicated MRL-trained bottleneck layer.** This is the cleanest approach if you're designing a new network or can modify the architecture.

```
Input → [Encoder] → h ∈ R^d → [MRL Bottleneck] → z ∈ R^d → [Decoder] → output
                                     ^
                                     |
                              Train with MRL loss
                              so z[1:m] is useful ∀m ∈ M
```

The bottleneck is a linear layer `d → d` trained with MRL. The decoder is also linear (for weight slicing) or has a projection adapter.

```python
class MRL_Bottleneck(nn.Module):
    """
    A bottleneck layer that produces MRL-trained representations.
    The full d-dim output is used normally, but z[1:m] is also
    independently useful for any m in nesting_list.
    """
    def __init__(self, dim: int, nesting_list: list[int],
                 num_classes: int, efficient: bool = True):
        super().__init__()
        self.dim = dim
        self.nesting_list = sorted(nesting_list)
        self.efficient = efficient

        # Bottleneck: d → d (can be identity if encoder already outputs d-dim)
        self.bottleneck = nn.Linear(dim, dim)

        # MRL classifier heads
        if efficient:
            self.classifier = nn.Linear(nesting_list[-1], num_classes, bias=False)
        else:
            self.classifiers = nn.ModuleDict({
                str(m): nn.Linear(m, num_classes, bias=False)
                for m in nesting_list
            })

    def forward(self, h, return_embeddings=False, return_logits=True):
        z = self.bottleneck(h)  # (batch, d)

        result = {}

        if return_embeddings:
            result['embedding'] = z
            result['nested_embeddings'] = {m: z[..., :m] for m in self.nesting_list}

        if return_logits:
            logits_list = []
            for m in self.nesting_list:
                if self.efficient:
                    logit = torch.matmul(
                        z[..., :m],
                        self.classifier.weight[:, :m].t()
                    )
                else:
                    logit = self.classifiers[str(m)](z[..., :m])
                logits_list.append(logit)
            result['logits'] = logits_list

        return result

    @staticmethod
    def mrl_loss(logits_list, labels, weights=None):
        if weights is None:
            weights = [1.0] * len(logits_list)
        return sum(w * nn.functional.cross_entropy(l, labels)
                   for w, l in zip(weights, logits_list))
```

### Full network with bottleneck

```python
class NetworkWithMRLBottleneck(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, bottleneck_dim=512,
                 num_classes=10, nesting_list=None):
        super().__init__()
        if nesting_list is None:
            nesting_list = [32, 64, 128, 256, 512]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim), nn.ReLU(),
        )
        self.mrl_bottleneck = MRL_Bottleneck(
            dim=bottleneck_dim,
            nesting_list=nesting_list,
            num_classes=num_classes,
            efficient=True,
        )
        # Downstream decoder uses the full bottleneck output
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, training=True):
        h = self.encoder(x)
        result = self.mrl_bottleneck(h, return_embeddings=True, return_logits=training)

        if training:
            # MRL auxiliary loss on the bottleneck
            result['decoder_logits'] = self.decoder(result['embedding'])
            return result
        else:
            # At inference, use full embedding through decoder
            return self.decoder(result['embedding'])

    def compute_loss(self, result, labels, decoder_weight=1.0):
        # MRL loss on bottleneck
        mrl_loss = self.mrl_bottleneck.mrl_loss(result['logits'], labels)
        # Main task loss through decoder
        main_loss = nn.functional.cross_entropy(result['decoder_logits'], labels)
        return mrl_loss + decoder_weight * main_loss
```

### Inference with compressed intermediate

```python
# Option A: Full pipeline (no compression)
logits = model(x, training=False)

# Option B: Use compressed bottleneck embedding for downstream tasks
h = model.encoder(x)
z = model.mrl_bottleneck.bottleneck(h)
z_compressed = z[..., :64]  # 8x smaller intermediate representation

# If downstream is linear with sliceable weights:
w = model.decoder[0].weight[:, :64]
h_next = torch.matmul(z_compressed, w.t()) + model.decoder[0].bias
h_next = torch.relu(h_next)
logits = model.decoder[1](h_next)

# Or if downstream is non-linear, use a projection adapter (Approach 2)
```

---

## Comparison of Approaches

| Approach | Extra Params | Downstream Must Be Linear? | Training Complexity | Inference Savings |
|----------|-------------|---------------------------|---------------------|-------------------|
| **1. Weight-Slicing** | 0 | Yes | Low (extra loss terms) | Embedding storage + downstream FLOPs |
| **2. Projection Adapter** | O(d²) per nest level | No | Medium | Embedding storage only |
| **3. Nested Dropout** | 0 | No | Low (no loss change) | Embedding storage + downstream FLOPs (with zeroed dims) |
| **4. MRL Bottleneck** | O(d²) (bottleneck) | No (for main path) | Medium (auxiliary loss) | Embedding storage + downstream FLOPs (if linear) |

---

## Practical Recommendations

1. **If the downstream layer is linear** → Use **Approach 1 (Weight-Slicing)**. It's the most faithful to MRL, adds zero parameters, and gives both storage and FLOPs savings.

2. **If the downstream layer is non-linear (attention, conv, etc.)** → Use **Approach 2 (Projection Adapter)** if you care about storage/communication. Use **Approach 3 (Nested Dropout)** if you want zero extra parameters and are OK with implicit optimization.

3. **If you're designing a new network** → Use **Approach 4 (MRL Bottleneck)**. Insert a dedicated bottleneck, train it with MRL auxiliary loss alongside the main task loss, and get the best of both worlds: the main path uses full-dim, but the bottleneck embedding is multi-granular for adaptive deployment.

4. **If you have a pipeline-parallel or distributed setup** → The intermediate representation is transmitted between devices. Any approach that reduces the transmitted dimension (1, 3, or 4) directly reduces communication bandwidth.

5. **If you want to reduce downstream FLOPs too** → You need the downstream layer to work with `m`-dim input, not `d`-dim. This requires either weight-slicing (Approach 1) or zero-padding (Approach 3, but FLOPs aren't saved unless the framework skips zeros). True FLOPs savings on downstream layers generally requires structured dimension reduction (changing the layer's input size), which is what Approach 1 achieves for linear layers.

---

## Key Insight

MRL on an intermediate layer is fundamentally about **information routing**: forcing the network to put the most important information in the first `m` dimensions so that truncating the rest doesn't lose much. The mechanism (weight-slicing, projection, dropout, or bottleneck) determines how the downstream layers handle the reduced input, but the training signal is the same: **a loss at each granularity that rewards good performance with only the first `m` dimensions.**

The paper doesn't explicitly test intermediate-layer MRL, but the principle is sound — it's the same optimization, just applied at a different point in the network. The main risk is that intermediate representations may be harder to compress than final embeddings, because they encode features rather than semantics, and the downstream layers may rely on specific dimensions in ways that are harder to reorganize.
