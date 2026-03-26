"""
Flask web app focused on AI-generated image detection.

Supports:
- Single-image analysis
- Batch analysis for up to 5 images
- Forensic visualizations (FFT and ELA)
"""

from flask import Flask, jsonify, render_template, request
from PIL import Image
import base64
import hashlib
import io
import json
import logging
import os
import threading
try:
    import redis
except ModuleNotFoundError:
    redis = None

import numpy as np

try:
    from web.history_store import build_history_store
except ModuleNotFoundError:
    from history_store import build_history_store


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB total request size
MAX_BATCH_IMAGES = 5
MAX_ANALYSIS_DIM = 512
MAX_HISTORY_ITEMS = 12
HISTORY_PREVIEW_DIM = 256
LOGGER = logging.getLogger(__name__)
HISTORY_STORE = build_history_store()

# Initialize Redis client for caching
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
if redis is None:
    LOGGER.info("Redis package not installed; caching disabled.")
    cache = None
else:
    try:
        cache = redis.from_url(REDIS_URL, decode_responses=True)
        cache.ping()
        LOGGER.info("Connected to Redis cache: %s", REDIS_URL)
    except Exception as exc:
        LOGGER.warning("Redis cache unavailable: %s", exc)
        cache = None


def get_image_hash(image_bytes):
    """Generate SHA-256 hash for image caching."""
    return hashlib.sha256(image_bytes).hexdigest()


def resize_for_analysis(image_pil):
    """Resize large images to keep CPU inference responsive."""
    if max(image_pil.size) <= MAX_ANALYSIS_DIM:
        return image_pil.copy()

    ratio = MAX_ANALYSIS_DIM / max(image_pil.size)
    new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
    return image_pil.resize(new_size, Image.LANCZOS)


def verdict_from_score(score):
    """Map ensemble score to a user-facing verdict."""
    if score >= 0.55:
        return "AI-Generated", "AI"
    if score >= 0.40:
        return "Likely AI-Generated", "Likely AI"
    if score >= 0.25:
        return "Uncertain", "Mixed"
    if score >= 0.10:
        return "Likely Real", "Likely Real"
    return "Real", "Real"


def image_to_data_url(image_pil):
    """Encode a PIL image as a PNG data URL."""
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def make_history_preview(image_pil):
    """Create a small preview for persisted analysis history."""
    preview = image_pil.copy()
    preview.thumbnail((HISTORY_PREVIEW_DIM, HISTORY_PREVIEW_DIM), Image.LANCZOS)
    return image_to_data_url(preview)


def analyze_fft(image_np):
    """
    FFT frequency-domain analysis.
    AI-generated images often show atypical spectral energy distribution.
    """
    try:
        gray = np.mean(image_np, axis=2) if len(image_np.shape) == 3 else image_np.copy()
        gray = gray.astype(np.float64)

        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        if magnitude.max() > 0:
            magnitude_norm = magnitude / magnitude.max()
        else:
            magnitude_norm = magnitude

        h, w = magnitude_norm.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_radius = max(1.0, dist.max())

        low_mask = dist <= max_radius * 0.15
        mid_mask = (dist > max_radius * 0.15) & (dist <= max_radius * 0.45)
        high_mask = dist > max_radius * 0.45

        total_energy = np.sum(magnitude_norm)
        if total_energy == 0:
            return 0.5, magnitude_norm, {"error": "Empty FFT energy"}

        low_energy = np.sum(magnitude_norm[low_mask]) / total_energy
        mid_energy = np.sum(magnitude_norm[mid_mask]) / total_energy
        high_energy = np.sum(magnitude_norm[high_mask]) / total_energy
        energy_ratio = low_energy / (high_energy + 1e-10)
        spectral_std = np.std(magnitude_norm)

        score = 0.0
        if high_energy < 0.12:
            score += 0.35
        elif high_energy < 0.18:
            score += 0.12 + 0.23 * ((0.18 - high_energy) / 0.06)
        elif high_energy < 0.24:
            score += 0.12 * ((0.24 - high_energy) / 0.06)

        if energy_ratio > 3.5:
            score += 0.30
        elif energy_ratio > 2.4:
            score += 0.10 + 0.20 * ((energy_ratio - 2.4) / 1.1)
        elif energy_ratio > 1.8:
            score += 0.10 * ((energy_ratio - 1.8) / 0.6)

        if spectral_std < 0.13:
            score += 0.20
        elif spectral_std < 0.18:
            score += 0.08 + 0.12 * ((0.18 - spectral_std) / 0.05)
        elif spectral_std < 0.22:
            score += 0.08 * ((0.22 - spectral_std) / 0.04)

        if high_energy > 0.20 and energy_ratio < 1.7:
            score -= 0.08

        score = min(1.0, max(0.0, score))

        details = {
            "low_freq_energy": round(float(low_energy), 4),
            "mid_freq_energy": round(float(mid_energy), 4),
            "high_freq_energy": round(float(high_energy), 4),
            "energy_ratio": round(float(energy_ratio), 4),
            "spectral_std": round(float(spectral_std), 6),
        }
        return score, magnitude_norm, details
    except Exception as exc:
        return 0.5, None, {"error": str(exc)}


def analyze_ela(image_pil):
    """
    Error Level Analysis.
    AI-generated images often show unusually uniform re-compression error.
    """
    try:
        buffer = io.BytesIO()
        image_pil.convert("RGB").save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        orig_np = np.array(image_pil.convert("RGB"), dtype=np.float64)
        recomp_np = np.array(recompressed, dtype=np.float64)
        ela_diff = np.abs(orig_np - recomp_np)
        ela_gray = np.mean(ela_diff, axis=2)
        ela_visual = np.clip(ela_gray * 20.0, 0, 255).astype(np.uint8)

        ela_mean = np.mean(ela_gray)
        ela_std = np.std(ela_gray)
        ela_max = np.max(ela_gray)
        cv = ela_std / (ela_mean + 1e-10)

        bh, bw = ela_gray.shape[0] // 8, ela_gray.shape[1] // 8
        if bh > 0 and bw > 0:
            block_means = []
            for i in range(8):
                for j in range(8):
                    block = ela_gray[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
                    block_means.append(np.mean(block))
            block_means = np.array(block_means)
            block_cv = np.std(block_means) / (np.mean(block_means) + 1e-10)
        else:
            block_cv = cv

        score = 0.0
        if ela_mean < 1.0:
            score += 0.35
        elif ela_mean < 2.5:
            score += 0.12 + 0.23 * ((2.5 - ela_mean) / 1.5)
        elif ela_mean < 4.5:
            score += 0.12 * ((4.5 - ela_mean) / 2.0)

        if cv < 0.50:
            score += 0.25
        elif cv < 0.70:
            score += 0.08 + 0.17 * ((0.70 - cv) / 0.20)
        elif cv < 1.0:
            score += 0.08 * ((1.0 - cv) / 0.3)

        if block_cv < 0.25:
            score += 0.25
        elif block_cv < 0.50:
            score += 0.08 + 0.17 * ((0.50 - block_cv) / 0.25)
        elif block_cv < 0.8:
            score += 0.08 * ((0.8 - block_cv) / 0.3)

        if ela_mean > 7.0 and cv > 1.4:
            score -= 0.10
        elif ela_mean > 4.5 and cv > 1.1:
            score -= 0.05

        score = min(1.0, max(0.0, score))
        details = {
            "ela_mean": round(float(ela_mean), 4),
            "ela_std": round(float(ela_std), 4),
            "ela_max": round(float(ela_max), 4),
            "coefficient_of_variation": round(float(cv), 4),
            "block_cv": round(float(block_cv), 4),
        }
        return score, ela_visual, details
    except Exception as exc:
        return 0.5, None, {"error": str(exc)}


def analyze_statistics(image_np):
    """
    Statistical feature analysis.
    """
    try:
        from scipy import stats as sp_stats

        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)

        img = image_np.astype(np.float64)
        channel_kurtosis = []
        channel_skewness = []
        hist_smoothness = []

        for c in range(3):
            channel = img[:, :, c].flatten()
            kurt = sp_stats.kurtosis(channel, fisher=True)
            skew = sp_stats.skew(channel)
            channel_kurtosis.append(float(kurt))
            channel_skewness.append(float(skew))

            hist, _ = np.histogram(channel, bins=64, range=(0, 255))
            hist_norm = hist / (hist.sum() + 1e-10)
            hist_diff2 = np.diff(hist_norm, n=2)
            roughness = np.sum(hist_diff2 ** 2)
            hist_smoothness.append(float(roughness))

        avg_kurtosis = np.mean(channel_kurtosis)
        avg_smoothness = np.mean(hist_smoothness)

        r = img[:, :, 0].flatten()
        g = img[:, :, 1].flatten()
        b = img[:, :, 2].flatten()
        rg_corr = np.corrcoef(r, g)[0, 1] if np.std(r) > 0 and np.std(g) > 0 else 0
        rb_corr = np.corrcoef(r, b)[0, 1] if np.std(r) > 0 and np.std(b) > 0 else 0
        gb_corr = np.corrcoef(g, b)[0, 1] if np.std(g) > 0 and np.std(b) > 0 else 0
        avg_correlation = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3.0

        noise_h = np.mean(np.abs(np.diff(img, axis=1)))
        noise_v = np.mean(np.abs(np.diff(img, axis=0)))
        noise_level = (noise_h + noise_v) / 2.0

        score = 0.0
        if avg_kurtosis < -0.5:
            score += 0.25
        elif avg_kurtosis < 0.5:
            score += 0.06 + 0.19 * ((0.5 - avg_kurtosis) / 1.0)
        elif avg_kurtosis < 1.5:
            score += 0.06 * ((1.5 - avg_kurtosis) / 1.0)

        if avg_smoothness < 0.0003:
            score += 0.45
        elif avg_smoothness < 0.0006:
            score += 0.20 + 0.25 * ((0.0006 - avg_smoothness) / 0.0003)
        elif avg_smoothness < 0.0008:
            score += 0.10 * ((0.0008 - avg_smoothness) / 0.0002)

        if avg_correlation > 0.98:
            score += 0.25
        elif avg_correlation > 0.95:
            score += 0.08 + 0.17 * ((avg_correlation - 0.95) / 0.03)
        elif avg_correlation > 0.90:
            score += 0.08 * ((avg_correlation - 0.90) / 0.05)

        if noise_level < 4.0:
            score += 0.20
        elif noise_level < 8.0:
            score += 0.06 + 0.14 * ((8.0 - noise_level) / 4.0)
        elif noise_level < 14.0:
            score += 0.06 * ((14.0 - noise_level) / 6.0)

        if noise_level > 18.0 or avg_kurtosis > 2.5:
            score -= 0.10
        elif noise_level > 12.0 or avg_kurtosis > 1.8:
            score -= 0.05

        score = min(1.0, max(0.0, score))
        details = {
            "avg_kurtosis": round(float(avg_kurtosis), 4),
            "avg_skewness": round(float(np.mean(channel_skewness)), 4),
            "histogram_smoothness": round(float(avg_smoothness), 6),
            "channel_correlation": round(float(avg_correlation), 4),
            "noise_level": round(float(noise_level), 4),
        }
        return score, details
    except Exception as exc:
        return 0.5, {"error": str(exc)}


def analyze_texture(image_np):
    """
    Texture consistency analysis using simplified LBP.
    """
    try:
        gray = np.mean(image_np, axis=2) if len(image_np.shape) == 3 else image_np.copy()
        gray = gray.astype(np.float64)
        h, w = gray.shape

        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (1 << 7) if gray[i - 1, j - 1] >= center else 0
                code |= (1 << 6) if gray[i - 1, j] >= center else 0
                code |= (1 << 5) if gray[i - 1, j + 1] >= center else 0
                code |= (1 << 4) if gray[i, j + 1] >= center else 0
                code |= (1 << 3) if gray[i + 1, j + 1] >= center else 0
                code |= (1 << 2) if gray[i + 1, j] >= center else 0
                code |= (1 << 1) if gray[i + 1, j - 1] >= center else 0
                code |= (1 << 0) if gray[i, j - 1] >= center else 0
                lbp[i - 1, j - 1] = code

        global_hist, _ = np.histogram(lbp.flatten(), bins=64, range=(0, 256))
        global_hist = global_hist / (global_hist.sum() + 1e-10)
        global_entropy = -np.sum(global_hist * np.log2(global_hist + 1e-10))

        lbp_h, lbp_w = lbp.shape
        bh, bw = max(1, lbp_h // 4), max(1, lbp_w // 4)
        block_entropies = []
        block_hists = []
        for i in range(4):
            for j in range(4):
                block = lbp[i * bh : min((i + 1) * bh, lbp_h), j * bw : min((j + 1) * bw, lbp_w)]
                if block.size == 0:
                    continue
                bk_hist, _ = np.histogram(block.flatten(), bins=64, range=(0, 256))
                bk_hist = bk_hist / (bk_hist.sum() + 1e-10)
                block_hists.append(bk_hist)
                block_entropies.append(-np.sum(bk_hist * np.log2(bk_hist + 1e-10)))

        if len(block_entropies) > 1:
            entropy_std = np.std(block_entropies)
            entropy_mean = np.mean(block_entropies)
            entropy_cv = entropy_std / (entropy_mean + 1e-10)
        else:
            entropy_cv = 0.5

        if len(block_hists) > 1:
            chi_sq_dists = []
            for i in range(len(block_hists)):
                for j in range(i + 1, len(block_hists)):
                    chi_sq = np.sum(
                        (block_hists[i] - block_hists[j]) ** 2
                        / (block_hists[i] + block_hists[j] + 1e-10)
                    )
                    chi_sq_dists.append(chi_sq)
            avg_chi_sq = np.mean(chi_sq_dists)
        else:
            avg_chi_sq = 0.5

        score = 0.0
        if entropy_cv < 0.05:
            score += 0.35
        elif entropy_cv < 0.12:
            score += 0.12 + 0.23 * ((0.12 - entropy_cv) / 0.07)
        elif entropy_cv < 0.20:
            score += 0.12 * ((0.20 - entropy_cv) / 0.08)

        if avg_chi_sq < 0.10:
            score += 0.30
        elif avg_chi_sq < 0.30:
            score += 0.10 + 0.20 * ((0.30 - avg_chi_sq) / 0.20)
        elif avg_chi_sq < 0.60:
            score += 0.10 * ((0.60 - avg_chi_sq) / 0.30)

        if global_entropy < 2.9:
            score += 0.20
        elif global_entropy < 3.9:
            score += 0.06 + 0.14 * ((3.9 - global_entropy) / 1.0)
        elif global_entropy < 4.8:
            score += 0.06 * ((4.8 - global_entropy) / 0.9)

        if entropy_cv > 0.22 and avg_chi_sq > 0.7:
            score -= 0.10
        elif entropy_cv > 0.16 or avg_chi_sq > 0.5:
            score -= 0.05

        score = min(1.0, max(0.0, score))
        details = {
            "global_entropy": round(float(global_entropy), 4),
            "entropy_cv": round(float(entropy_cv), 4),
            "avg_chi_sq_distance": round(float(avg_chi_sq), 4),
            "num_blocks": len(block_entropies),
        }
        return score, details
    except Exception as exc:
        return 0.5, {"error": str(exc)}


def analyze_image(image_pil, image_bytes=None):
    """Run all detectors and return a serialized response object."""
    cache_key = None
    if cache and image_bytes:
        cache_key = f"analysis:{get_image_hash(image_bytes)}"
        try:
            cached_result = cache.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result["cached"] = True
                return result
        except Exception as exc:
            LOGGER.warning("Redis cache read failed: %s", exc)

    image_resized = resize_for_analysis(image_pil)
    image_np = np.array(image_resized, dtype=np.float64)

    fft_score, fft_magnitude, fft_details = analyze_fft(image_np)
    ela_score, ela_visual, ela_details = analyze_ela(image_resized)
    stat_score, stat_details = analyze_statistics(image_np)
    texture_score, texture_details = analyze_texture(image_np)

    weights = {
        "fft": 0.25,
        "ela": 0.30,
        "statistics": 0.25,
        "texture": 0.20,
    }
    ensemble_score = (
        fft_score * weights["fft"]
        + ela_score * weights["ela"]
        + stat_score * weights["statistics"]
        + texture_score * weights["texture"]
    )
    ensemble_score = min(1.0, max(0.0, ensemble_score))

    if "histogram_smoothness" in stat_details:
        smoothness = float(stat_details["histogram_smoothness"])
        if smoothness < 0.0005:
            ensemble_score += 0.15
        elif smoothness > 0.0008:
            ensemble_score -= 0.15
    ensemble_score = min(1.0, max(0.0, ensemble_score))

    verdict, verdict_short = verdict_from_score(ensemble_score)

    fft_data_url = ""
    if fft_magnitude is not None:
        fft_vis = (fft_magnitude * 255).astype(np.uint8)
        fft_img = Image.fromarray(fft_vis).resize((256, 256), Image.LANCZOS)
        fft_data_url = image_to_data_url(fft_img)

    ela_data_url = ""
    if ela_visual is not None:
        ela_img = Image.fromarray(ela_visual).resize((256, 256), Image.LANCZOS)
        ela_data_url = image_to_data_url(ela_img)

    result = {
        "verdict": verdict,
        "verdict_short": verdict_short,
        "ensemble_score": round(float(ensemble_score), 4),
        "detectors": {
            "fft": {
                "name": "FFT Frequency Analysis",
                "score": round(float(fft_score), 4),
                "weight": weights["fft"],
                "description": "Analyzes frequency-domain energy distribution.",
                "details": fft_details,
            },
            "ela": {
                "name": "Error Level Analysis",
                "score": round(float(ela_score), 4),
                "weight": weights["ela"],
                "description": "Compares JPEG re-compression error patterns.",
                "details": ela_details,
            },
            "statistics": {
                "name": "Statistical Analysis",
                "score": round(float(stat_score), 4),
                "weight": weights["statistics"],
                "description": "Evaluates pixel histograms and channel correlation.",
                "details": stat_details,
            },
            "texture": {
                "name": "Texture Consistency",
                "score": round(float(texture_score), 4),
                "weight": weights["texture"],
                "description": "Checks local texture pattern uniformity.",
                "details": texture_details,
            },
        },
        "visualizations": {
            "fft_spectrum": fft_data_url,
            "ela_heatmap": ela_data_url,
        },
        "image_info": {
            "original_size": list(image_pil.size),
            "analyzed_size": list(image_resized.size),
        },
        "cached": False,
    }

    if cache and cache_key:
        try:
            # Cache for 24 hours
            cache.setex(cache_key, 86400, json.dumps(result))
        except Exception as exc:
            LOGGER.warning("Redis cache write failed: %s", exc)

    return result


def build_history_record(filename, image_pil, analysis_result):
    """Create a lightweight persistence record for one analysis."""
    return {
        "filename": filename,
        "verdict": analysis_result["verdict"],
        "verdict_short": analysis_result["verdict_short"],
        "ensemble_score": analysis_result["ensemble_score"],
        "input_preview_url": make_history_preview(image_pil),
        "fft_spectrum_url": analysis_result["visualizations"]["fft_spectrum"],
        "ela_heatmap_url": analysis_result["visualizations"]["ela_heatmap"],
        "image_info": analysis_result["image_info"],
        "detector_scores": {
            "fft": analysis_result["detectors"]["fft"]["score"],
            "ela": analysis_result["detectors"]["ela"]["score"],
            "statistics": analysis_result["detectors"]["statistics"]["score"],
            "texture": analysis_result["detectors"]["texture"]["score"],
        },
    }


def persist_history_async(records):
    """Persist history in the background so analysis latency stays low."""
    if not HISTORY_STORE.is_enabled() or not records:
        return

    def worker():
        try:
            HISTORY_STORE.save_many(records)
        except Exception as exc:
            LOGGER.warning("History persistence worker failed: %s", exc)

    threading.Thread(target=worker, daemon=True).start()


def analyze_file_storage(file_storage):
    """Load and analyze one uploaded file."""
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = analyze_image(image, image_bytes=image_bytes)
    filename = file_storage.filename or "image"
    result["filename"] = filename
    result["_history_record"] = build_history_record(filename, image, result)
    return result


@app.route("/")
def index():
    """Render the AI detector interface."""
    return render_template("index.html", max_batch_images=MAX_BATCH_IMAGES)


@app.route("/api/detect_ai", methods=["POST"])
def detect_ai():
    """Analyze a single uploaded image."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        result = analyze_file_storage(request.files["image"])
        history_record = result.pop("_history_record", None)
        persist_history_async([history_record] if history_record else [])
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/detect_ai_batch", methods=["POST"])
def detect_ai_batch():
    """Analyze up to MAX_BATCH_IMAGES in one request."""
    try:
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No images uploaded"}), 400
        if len(files) > MAX_BATCH_IMAGES:
            return jsonify({"error": f"Upload up to {MAX_BATCH_IMAGES} images at a time"}), 400

        results = [analyze_file_storage(file_storage) for file_storage in files]
        history_records = []
        for result in results:
            history_record = result.pop("_history_record", None)
            if history_record:
                history_records.append(history_record)
        persist_history_async(history_records)
        return jsonify({"count": len(results), "results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return recent persisted analyses when history is enabled."""
    if not HISTORY_STORE.is_enabled():
        return jsonify({"enabled": False, "backend": HISTORY_STORE.backend_name, "results": []})

    limit = request.args.get("limit", default=MAX_HISTORY_ITEMS, type=int)
    results = HISTORY_STORE.list_recent(limit=limit)
    return jsonify(
        {
            "enabled": True,
            "backend": HISTORY_STORE.backend_name,
            "count": len(results),
            "results": results,
        }
    )


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return frontend capability flags."""
    return jsonify(
        {
            "mode": "ai_detection_only",
            "max_batch_images": MAX_BATCH_IMAGES,
            "max_request_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            "history_enabled": HISTORY_STORE.is_enabled(),
            "history_backend": HISTORY_STORE.backend_name,
            "max_history_items": MAX_HISTORY_ITEMS,
            "features": [
                "single-image AI detection",
                "batch AI detection",
                "FFT visualization",
                "ELA visualization",
            ],
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("AI Image Detection Web App")
    print("=" * 60)
    print(f"Batch size limit: {MAX_BATCH_IMAGES} images")
    port = int(os.environ.get("PORT", 5000))
    print(f"Open your browser and go to: http://localhost:{port}")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port)
