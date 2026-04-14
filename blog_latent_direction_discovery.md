# Exploring Latent Directions in Video Diffusion Models

*April 2026*

---

What if you could steer the output of a video generation model the way you adjust brightness on your phone? Not by retraining, not by rewriting prompts — just by nudging a vector in the model's internal representation. That idea, which made GAN-based image editing almost magical a few years ago, is what got us exploring the latent space of **LTX-2**, a modern video diffusion model. The results so far are early — some things worked, some didn't, and the "why" behind both is genuinely interesting.

This post is an informal walkthrough of what we observed, what surprised us, and where we think this line of investigation might go.

---

## The Idea: Directions in Latent Space

In the GAN era, one of the cleanest discoveries was that a model's internal latent space often contains **linear directions** tied to human-understandable concepts. Move a certain direction in StyleGAN's latent space and you get "older face" or "more smile." No model retraining required — just vector arithmetic.

LTX-2 compresses video through a VAE (Variational Autoencoder) before its diffusion transformer processes it. That compressed representation — the latent space — becomes the natural first place to look for similar direction-based editing. The question we wanted to answer: *does the latent space of a video diffusion model's VAE hold interpretable, steerable directions?*

---

## What We Tried

We tested a straightforward approach: take a video, apply a known pixel-space transformation (brightness reduction, contrast shift, grayscale conversion, zoom), encode both the original and transformed version, then compute the **difference vector** in latent space.

```
direction = encode(transformed_video) - encode(original_video)
```

This gives us a candidate "direction" in latent space that should, in theory, correspond to that visual transformation. We then tested whether adding this direction back to *other* videos would reproduce a similar effect — the key test of whether these are genuine semantic axes versus encoding artifacts.

We swept across **10 different strengths** (alpha values from 0.5 to 2.5) and tested on four different source videos, giving us a fairly broad picture of how the approach behaves.

Here's a look at one of our source videos, and the pixel-space reference transformation we were trying to learn a latent direction for:

<table>
<tr>
<td align="center"><b>Original source video</b></td>
<td align="center"><b>Reference: brightness darkened in pixel space</b></td>
</tr>
<tr>
<td><video src="references/source_preprocessed.mp4" width="320" controls></video></td>
<td><video src="references/reference_brightness_reduced.mp4" width="320" controls></video></td>
</tr>
</table>

---

## What We Observed

### The effects are subtle — but they exist

When we applied the brightness direction at moderate strength (alpha ~2.0), the result wasn't a dramatic darkening — it was a gentle, visible shift. Compared to the 40% darkening we used to *derive* the direction, the latent approach reproduced roughly a fifth of that. Subtle, but detectably real.

Here's what the latent-space brightness direction produces at alpha=2.0, alongside a 20x-amplified difference map that makes the changes easier to see:

<table>
<tr>
<td align="center"><b>Brightness direction applied (α=2.0)</b></td>
<td align="center"><b>Difference from original (amplified 20×)</b></td>
</tr>
<tr>
<td><video src="videos/8154896-sd_960_540_25fps_brightness_am2_000.mp4" width="320" controls></video></td>
<td><video src="videos/8154896-sd_960_540_25fps_brightness_am2_000_diff_x20.mp4" width="320" controls></video></td>
</tr>
</table>

The amplified diff map is interesting — you can see the direction isn't just applying a flat brightness shift. The changes are spatially structured and follow the content of the video, which suggests the latent space *is* encoding something about the visual structure, even if the edits aren't as strong as we'd like.

### Some directions respond better than others

Not all transformations were equal. Brightness produced the most measurable effect, while contrast had a moderate response. Grayscale and zoom directions produced very faint changes — close to what we saw from purely random directions. This is actually informative: it hints that the latent space might have a **non-uniform sensitivity** to different visual properties. Simple global transforms like brightness might partially align with how the VAE organizes its encoding, while geometric transforms like zoom might be encoded in a way that's more tangled with content.

### Alpha scaling works, but has a ceiling

Increasing alpha did produce stronger effects, which is a good sign — it means the direction isn't pure noise. But the useful range is narrow. Below alpha 1.0, changes are barely measurable. Above 2.5, we start seeing decode artifacts before the intended transformation becomes really crisp. There's a window of useful operation, but it's more of a peephole than a bay window.

### Transfer across videos is inconsistent

A direction derived from one video applied to another doesn't always produce the same strength or quality of effect. Some videos respond well to a given brightness direction; others barely react at all. This suggests the latent geometry isn't globally uniform — local neighborhoods in the space may have different structure. It's a challenge, but not necessarily a dead end: GAN directions had similar issues early on before methods like GANSpace found more robust principal directions.

---

## Why This Is Hard (and Why It's Still Interesting)

The core observation is that the VAE latent space in diffusion models serves a different purpose than the latent space in GANs. In a GAN, the latent space *is* the generation bottleneck — all the model's learned structure flows through it. In a diffusion model, the VAE is primarily a **compression stage**, designed to make the diffusion process computationally tractable. The semantic heavy lifting happens inside the diffusion transformer, conditioned on text prompts and attention mechanisms.

So the VAE latent space isn't *designed* to have clean semantic directions. But that doesn't mean they can't emerge. Compression still requires the encoder to organize visual information in a structured way, and some of that structure might be usable — we just need better tools to find and exploit it.

The other complication is video. A direction needs to produce coherent changes across time, space, and channels simultaneously. The latent tensor for a video isn't a simple vector — it's a high-dimensional object with temporal dependencies. This makes the search space much larger and the problem genuinely harder than image-domain direction discovery.

---

## Where This Could Go

Several promising directions emerge from these observations:

**Better direction discovery methods.** We used simple difference vectors — the most straightforward approach. Methods like PCA over many videos, contrastive learning of direction pairs, or even learning directions via lightweight optimization could find more robust axes. The space may have useful structure that simple subtraction can't extract.

**Hybrid approaches.** Rather than editing in the VAE latent space alone, combining latent perturbations with attention manipulation in the diffusion transformer could amplify weak latent signals. The transformer could act as a semantic amplifier for directions that the VAE encodes weakly.

**Region-adaptive directions.** Since transfer across videos is inconsistent, video-specific or region-specific direction calibration might help. A small per-video projection step could map a global direction to the local latent geometry around that video's encoding.

**Noise-space directions.** Instead of perturbing the VAE's latent output, editing the initial noise fed into the diffusion process might better leverage the denoiser's semantic understanding. The denoiser was trained to extract structure from noise — perturbations in noise space might flow through the semantic bottleneck more effectively than post-hoc latent edits.

**Activation-space editing.** The diffusion transformer's internal activations are where semantic decisions actually get made. Methods from the mechanistic interpretability community — activation patching, causal tracing — could identify meaningful directions inside the denoiser itself, where the model's visual understanding is richest.

---

## Takeaways

The VAE latent space of LTX-2 isn't a clean semantic remote control — but it isn't a featureless void either. There are real, measurable signals when you push in the right direction, especially for global visual properties like brightness. The effects are weaker than what the GAN world trained us to expect, and transfer across videos remains an open challenge.

What excites us is that this is still very early. The GAN direction-discovery literature went through years of refinement — from random probing to InterFaceGAN to GANSpace to StyleCLIP — each step extracting more usable structure from the same latent spaces. Video diffusion models are architecturally different and arguably harder, but the core question — *can we find intuitive control surfaces inside generative models?* — remains as compelling as ever.

If the semantic structure isn't (fully) in the VAE, it's somewhere in the model. Finding it is the next step.
