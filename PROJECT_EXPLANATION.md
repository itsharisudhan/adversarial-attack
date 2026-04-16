# 🛡️ Project Explanation: Unified Fraud Detection System
> A simple, point-wise breakdown of what we built and how it works.

---

### 🌟 1. What is this project?
In simple terms, it's a **Digital Truth-Checker**. 
* When someone uploads an image (like an ID or a receipt), our system checks if they are trying to cheat.
* It looks for hidden changes, AI-generated fakes, and repeated uploads.
* Instead of trusting just one tool, it uses **three different scanners** (layers) to give a final verdict.

---

### 📂 2. How we built it? (The Tech Stack)
We didn't just write a script; we built a full system:
*   **The Brain (AI):** We used **PyTorch** to run deep-learning models that "look" at the images.
*   **The Website (Frontend/Backend):** Built with **Flask** (Python) so users can easily upload images and see results in real-time.
*   **The Memory (Database):** We used **Supabase (PostgreSQL)** to save every analysis, so we have a history of all threats.
*   **The Library:** We used **FAISS** (by Facebook) to store a "library" of normal images to compare new uploads against.

---

### 🧠 3. The 3 Main Algorithms (The "Heads")

#### **A. The Adversarial Head (Finding "Invisble Noise")**
*   **Algorithm:** **LID (Local Intrinsic Dimensionality)** & **Mahalanobis Distance**.
*   **How it works:** Hackers add invisible noise to images to trick AI. It looks normal to humans but "vibrates" differently to a computer. These algorithms measure that "vibration" to see if the image sits outside the "normal" range.
*   **In short:** It catches sophisticated hacks that try to bypass standard security.

#### **B. The Forensic Head (Finding "Digital Scars")**
*   **Algorithm:** **FFT (Fast Fourier Transform)** & **ELA (Error Level Analysis)**.
*   **How it works:** 
    *   **ELA** looks for inconsistencies in how much an image has been compressed (edited parts look different).
    *   **FFT** looks for mathematical patterns found only in AI-generated images (like Stable Diffusion).
*   **In short:** It catches "Photoshopped" images and AI Deepfakes.

#### **C. The Duplicate Head (Finding "Copy-Cats")**
*   **Algorithm:** **SHA-256 Hashing** & **Cosine Similarity**.
*   **How it works:** 
    *   **Hashing** gives every image a unique ID. If two IDs match, it's an exact copy.
    *   **Cosine Similarity** calculates an angle between two images in a multi-dimensional space. If the angle is tiny, the images are almost identical (even if one is resized or cropped).
*   **In short:** It stops people from using one stolen photo 100 times.

---

### ⚙️ 4. How the "Unified Verdict" works?
After the 3 heads scan the image, they report back to the **Master Decision Maker**:
1.  **Duplicate check first:** if the image has been seen before, it's flagged as `DUPLICATE_FRAUD` immediately.
2.  **Score Fusion:** It combines the "vibe" (adversarial score) and the "scars" (forensic score).
3.  **The Result:** 
    *   If many signs point to AI → `AI_GENERATED`.
    *   If hidden noise is detected → `ADVERSARIAL_ATTACK`.
    *   If part of it looks edited → `TAMPERED`.
    *   If everything is clean → `CLEAN`.

---

### 🚀 5. Why is this project better?
Most systems only look for **one** thing (like just Deepfakes). **Our system is "Unified."** 
It's like having a security team where: 
* One person looks at the ID under UV light (**Forensics**).
* One person checks if the ID was stolen (**Adversarial**).
* One person checks if they've seen this person before (**Duplicates**).

Together, they are much harder to trick!

---
