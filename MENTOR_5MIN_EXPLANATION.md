# 5-Minute Mentor Explanation

## Project Title

**Unified Adversarial and GenAI Detection System**

## 5-Minute Technical Explanation

This project is a **Unified Adversarial and GenAI Detection System** for image authenticity verification.

The main goal of the project is to determine whether an uploaded image is genuine, AI-generated, adversarially manipulated, tampered with, or reused as a duplicate.

From a technical point of view, the system has a **Flask backend** and a **web interface** where the user uploads images and gets analysis results.

The detection pipeline works in **three main layers**.

### 1. Forensic Analysis Layer

The first layer performs image forensics using:

- FFT analysis
- ELA or Error Level Analysis
- statistical feature analysis
- texture analysis

This layer is mainly used to identify signs of **GenAI-generated content** or **tampering artifacts**.

### 2. Adversarial Detection Layer

The second layer checks whether the image has been adversarially manipulated.

For this, the system extracts deep image embeddings using **EfficientNet-B0**, then applies:

- **LID** or Local Intrinsic Dimensionality
- **Mahalanobis distance**

These methods measure how far the input image is from the distribution of clean reference images. If the image lies far from the normal image manifold, it is treated as suspicious or adversarial.

### 3. Duplicate Fraud Detection Layer

The third layer checks whether the same image has been submitted before.

It uses:

- **SHA-256 hashing** for exact duplicate matching
- **cosine similarity on embeddings** for near-duplicate detection

This helps identify fraud cases where the same or slightly modified image is reused.

## Final Decision

All three layers are combined in a **unified detector** that produces a final verdict such as:

- `CLEAN`
- `AI_GENERATED`
- `ADVERSARIAL_ATTACK`
- `TAMPERED`
- `DUPLICATE_FRAUD`
- `HYBRID_THREAT`

So instead of depending on a single detector, the system fuses multiple signals to make the final decision more reliable.

## Database and Storage

The main persistent database used in this project is **Supabase/PostgreSQL**.

It is used to store:

- analysis history
- verdicts
- detector scores
- timestamps
- image metadata

We use PostgreSQL because the stored results are structured, queryable, and some fields such as detector outputs are flexible enough to be stored in `jsonb`.

In addition to that:

- **Redis** is used as an optional cache to speed up repeated image analysis
- **FAISS + NumPy checkpoint files** are used as the reference embedding database for adversarial detection

## Final Closing Line

So in short, this project extends a normal adversarial detection system into a practical **image trust and fraud detection platform** by combining **GenAI detection, adversarial detection, tamper analysis, and duplicate checking** in one unified system.

---

## 1-Minute Short Version

My project is a unified image verification system. It checks whether an image is real, AI-generated, adversarially manipulated, tampered with, or duplicated. Technically, it combines forensic analysis, embedding-based anomaly detection, and duplicate matching inside a Flask-based web system. For persistence, it uses Supabase/PostgreSQL, Redis for caching, and FAISS-based reference embeddings for adversarial detection.

---

## If Mentor Asks: What Is the Novelty?

The novelty of the project is that it does not rely on a single detection technique. Instead, it combines multiple complementary detection layers and fuses them into one final verdict, which makes the system more practical and robust for real-world image authenticity verification.
