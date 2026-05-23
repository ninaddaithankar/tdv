# TDV Difference Encoder with Cross-Attention

The **Temporal Difference Vision (TDV)** model learns video representations by jointly training two complementary encoders:
- a **Frame Encoder** that aims to capture *appearance and semantics*, and  
- a **Motion Encoder** that aims to capture *temporal change and dynamics* between frames.

Instead of reconstructing pixels, TDV predicts how latent embeddings evolve over time — learning to represent motion as *additive changes* in semantic space.

---

## Model Overview

### 1. Frame Encoder
A Vision Transformer backbone (e.g., DINOv2, MAE, or VAE architecture) trained **from scratch** to encode the semantic content of each frame.  
- **Input:** raw RGB frame  
- **Output:** embedding tensor `[num_frames, num_patches + 1, D]`  
- Learns static visual semantics, spatial structure, and context.  
- When EMA is enabled, a teacher copy of this encoder provides stable targets.

### 2. Motion Encoder
A transformer with **cross-attention** that encodes *RGB differences* between consecutive frames, conditioned on the previous frame’s embedding.  
It learns to represent how semantic embeddings should change over time:
`ΔE_t = f_motion(ΔRGB_t, E_t)`
This captures temporal dynamics — motion, object displacement, and state change — complementary to the frame encoder.

### 3. Linear Projection
A small linear layer aligns the motion encoder output to the same dimension as the frame encoder embedding, allowing additive composition:
`Ê_{t+1} = E_t + ΔE_t`

### 4. DINO Head (Optional)
A projection MLP used for self-distillation and contrastive alignment (DINO or iBOT objectives).  
It encourages invariance and stable feature representations across time and views.

### 5. EMA Teacher (Optional)
An exponential moving average of the frame encoder (and optionally the DINO head) provides a smoother, slowly-updating teacher target for distillation losses.

---

## Data Flow

1. Encode frames → frame embeddings `E_t`  
2. Compute RGB difference between frames → `ΔRGB_t`  
3. Encode motion conditioned on previous embedding → `ΔE_t = motion_encoder(ΔRGB_t | E_t)`  
4. Project `ΔE_t` and add to `E_t` → predicted embedding `Ē_{t+1}`  
5. Compare `Ē_{t+1}` to true or teacher-encoded `E_{t+1}` using multi-objective loss.

---

## Losses

The total loss is a weighted combination of complementary objectives:

| Loss | Description |
|------|--------------|
| **MSE Loss** | Reconstructs the next embedding directly in latent space. Encourages accurate temporal prediction. |
| **DINO / iBOT Loss** | Self-distillation on CLS and patch tokens, aligning student and teacher embeddings for semantic consistency. |
| **Motion Loss** | Ensures that embedding change magnitude reflects actual pixel-space motion, promoting physically meaningful dynamics. |

`L_total = λ_mse * L_mse + λ_dino * L_dino + λ_ibot * L_ibot + λ_motion * L_motion`

Each weight is tunable via hyperparameters.

---

## Summary

TDV learns **semantic-temporal decomposition** in a single architecture:
- The **Frame Encoder** grounds features in *visual semantics*.  
- The **Motion Encoder** models *temporal evolution* via cross-attention.  
- Additive latent updates (`E_t + ΔE_t`) enable stable, interpretable temporal prediction.  
- Self-distillation and motion regularization jointly ensure smooth, consistent, and causally grounded representation learning.
