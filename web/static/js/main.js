const MAX_BATCH_IMAGES = window.__AI_APP_CONFIG__.maxBatchImages;

let selectedFiles = [];
let loadingIntervalId = null;

document.addEventListener("DOMContentLoaded", () => {
    bindUI();
    loadStats();
});

function bindUI() {
    const upload = document.getElementById("aiImageUpload");
    const dropZone = document.getElementById("aiDropZone");

    upload.addEventListener("change", (event) => handleFileSelection(event.target.files));
    document.getElementById("btnAnalyzeBatch").addEventListener("click", analyzeBatch);
    document.getElementById("btnClearSelection").addEventListener("click", clearSelection);

    dropZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-active");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-active");
    });

    dropZone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-active");
        handleFileSelection(event.dataTransfer.files);
    });
}

async function loadStats() {
    try {
        const response = await fetch("/api/stats");
        const data = await response.json();
        document.getElementById("appStatus").textContent = `Ready`;
        const advPill = document.getElementById("advHeadPill");
        if (advPill) {
            advPill.textContent = data.adversarial_head_ready
                ? "Adversarial head: READY"
                : "Adversarial head: forensic-only";
            advPill.classList.toggle("pill-ready", data.adversarial_head_ready);
        }
    } catch (error) {
        document.getElementById("appStatus").textContent = "Offline";
    }
}

function handleFileSelection(fileList) {
    const files = Array.from(fileList).filter((file) => file.type.startsWith("image/"));
    selectedFiles = files.slice(0, MAX_BATCH_IMAGES);

    if (files.length > MAX_BATCH_IMAGES) {
        showNotification(`Only the first ${MAX_BATCH_IMAGES} images were kept.`, "info");
    }

    document.getElementById("aiImageUpload").value = "";
    renderPreviews();
    clearResults();
}

function clearSelection() {
    selectedFiles = [];
    renderPreviews();
    clearResults();
}

function clearResults() {
    document.getElementById("resultsStack").innerHTML = "";
    document.getElementById("summaryPanel").hidden = true;
    document.getElementById("summaryGrid").innerHTML = "";
}

function renderPreviews() {
    const previewGrid = document.getElementById("previewGrid");
    const selectionCount = document.getElementById("selectionCount");

    previewGrid.innerHTML = "";
    if (selectedFiles.length === 0) {
        selectionCount.textContent = "No images selected";
        return;
    }

    selectionCount.textContent = `${selectedFiles.length} image${selectedFiles.length === 1 ? "" : "s"} selected`;

    selectedFiles.forEach((file) => {
        const card = document.createElement("article");
        card.className = "preview-card";

        const img = document.createElement("img");
        img.alt = file.name;
        img.src = URL.createObjectURL(file);
        img.onload = () => URL.revokeObjectURL(img.src);

        const meta = document.createElement("div");
        meta.className = "preview-meta";
        meta.innerHTML = `
            <strong>${escapeHtml(file.name)}</strong>
            <span>${formatFileSize(file.size)}</span>
        `;

        card.appendChild(img);
        card.appendChild(meta);
        previewGrid.appendChild(card);
    });
}

async function analyzeBatch() {
    if (selectedFiles.length === 0) {
        showNotification("Select at least one image first.", "error");
        return;
    }

    setAnalyzeButtonState(true);
    showLoading(`Analyzing ${selectedFiles.length} image${selectedFiles.length === 1 ? "" : "s"} with unified detection...`);

    try {
        // Send each image to the unified endpoint
        const results = [];
        for (const file of selectedFiles) {
            const formData = new FormData();
            formData.append("image", file);

            const response = await fetch("/api/detect_unified", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `Analysis failed for ${file.name}`);
            }

            data.filename = file.name;
            results.push(data);
        }

        renderSummary(results);
        renderResults(results);
        showNotification("Analysis complete.", "success");
    } catch (error) {
        showNotification(error.message, "error");
    } finally {
        hideLoading();
        setAnalyzeButtonState(false);
    }
}

// ---- Summary ----

function renderSummary(results) {
    const summaryPanel = document.getElementById("summaryPanel");
    const summaryGrid = document.getElementById("summaryGrid");

    let cleanCount = 0, aiCount = 0, advCount = 0, dupCount = 0, tamperCount = 0, hybridCount = 0;

    results.forEach((r) => {
        const v = r.verdict || r.unified_verdict || "CLEAN";
        if (v === "CLEAN") cleanCount++;
        else if (v === "AI_GENERATED") aiCount++;
        else if (v === "ADVERSARIAL_ATTACK") advCount++;
        else if (v === "DUPLICATE_FRAUD") dupCount++;
        else if (v === "TAMPERED") tamperCount++;
        else if (v === "HYBRID_THREAT") hybridCount++;
    });

    summaryGrid.innerHTML = `
        <div class="summary-card">
            <span class="summary-label">Total analyzed</span>
            <strong>${results.length}</strong>
        </div>
        <div class="summary-card summary-clean">
            <span class="summary-label">Clean</span>
            <strong>${cleanCount}</strong>
        </div>
        <div class="summary-card summary-ai">
            <span class="summary-label">AI Generated</span>
            <strong>${aiCount}</strong>
        </div>
        <div class="summary-card summary-adv">
            <span class="summary-label">Adversarial</span>
            <strong>${advCount}</strong>
        </div>
        <div class="summary-card summary-dup">
            <span class="summary-label">Duplicate</span>
            <strong>${dupCount}</strong>
        </div>
        <div class="summary-card summary-tamper">
            <span class="summary-label">Tampered / Hybrid</span>
            <strong>${tamperCount + hybridCount}</strong>
        </div>
    `;

    summaryPanel.hidden = false;
}

// ---- Result Cards ----

function renderResults(results) {
    const resultsStack = document.getElementById("resultsStack");
    resultsStack.innerHTML = "";

    results.forEach((result, index) => {
        const card = document.createElement("section");
        card.className = "panel result-card";
        card.style.animationDelay = `${index * 0.08}s`;

        const verdict = result.verdict || result.unified_verdict || "CLEAN";
        const verdictClass = getVerdictClass(verdict);
        const confidence = result.confidence || 0;
        const advScore = result.adversarial_score || 0;
        const genaiScore = result.genai_score || 0;
        const isDuplicate = result.is_duplicate || false;
        const details = result.details || {};

        // Forensic scores
        const forensic = details.forensic_detectors || {};
        const fft = result.detectors?.fft || null;
        const ela = result.detectors?.ela || null;
        const stats = result.detectors?.statistics || null;
        const texture = result.detectors?.texture || null;

        card.innerHTML = `
            <div class="result-head">
                <div>
                    <p class="section-label">Result ${index + 1}</p>
                    <h2>${escapeHtml(result.filename || "Image")}</h2>
                </div>
                <div class="verdict-chip ${verdictClass}">${formatVerdict(verdict)}</div>
            </div>

            <!-- Unified Scores -->
            <div class="score-row">
                <div class="score-item">
                    <div class="score-bar-wrap">
                        <div class="score-bar-label">
                            <span>Confidence</span>
                            <strong>${(confidence * 100).toFixed(1)}%</strong>
                        </div>
                        <div class="verdict-bar">
                            <div class="verdict-bar-fill ${verdictClass}" style="width:${(confidence * 100).toFixed(1)}%"></div>
                        </div>
                    </div>
                </div>
                <div class="score-item">
                    <div class="score-bar-wrap">
                        <div class="score-bar-label">
                            <span>Adversarial Score</span>
                            <strong>${(advScore * 100).toFixed(1)}%</strong>
                        </div>
                        <div class="verdict-bar">
                            <div class="verdict-bar-fill is-adversarial" style="width:${(advScore * 100).toFixed(1)}%"></div>
                        </div>
                    </div>
                </div>
                <div class="score-item">
                    <div class="score-bar-wrap">
                        <div class="score-bar-label">
                            <span>GenAI Score</span>
                            <strong>${(genaiScore * 100).toFixed(1)}%</strong>
                        </div>
                        <div class="verdict-bar">
                            <div class="verdict-bar-fill is-ai" style="width:${(genaiScore * 100).toFixed(1)}%"></div>
                        </div>
                    </div>
                </div>
            </div>

            ${isDuplicate ? '<div class="duplicate-warning">This image was previously submitted!</div>' : ''}

            <!-- Manifold Metrics -->
            ${details.lid_score != null ? `
            <div class="metric-grid manifold-metrics">
                <article class="metric-panel">
                    <span class="metric-panel-label">Manifold</span>
                    <h3>LID Score</h3>
                    <p>Local Intrinsic Dimensionality. Higher = farther from clean image manifold.</p>
                    <div class="mini-score-row">
                        <span>${Number(details.lid_score).toFixed(2)}</span>
                        <span>Threshold: ${details.lid_threshold || 60}</span>
                    </div>
                </article>
                <article class="metric-panel">
                    <span class="metric-panel-label">Manifold</span>
                    <h3>Mahalanobis Distance</h3>
                    <p>Statistical distance from the centroid of clean images.</p>
                    <div class="mini-score-row">
                        <span>${Number(details.mahalanobis_distance).toFixed(2)}</span>
                        <span>Higher = more suspicious</span>
                    </div>
                </article>
            </div>
            ` : ''}

            <!-- Forensic Detectors -->
            ${fft || ela || stats || texture ? `
            <div class="metric-grid">
                ${fft ? renderDetectorCard("FFT", fft) : ''}
                ${ela ? renderDetectorCard("ELA", ela) : ''}
                ${stats ? renderDetectorCard("Stats", stats) : ''}
                ${texture ? renderDetectorCard("Texture", texture) : ''}
            </div>
            ` : ''}

            <!-- Visualizations -->
            ${result.visualizations ? `
            <div class="visual-grid">
                ${renderVisualCard("FFT Spectrum", result.visualizations.fft_spectrum)}
                ${renderVisualCard("ELA Heatmap", result.visualizations.ela_heatmap)}
            </div>
            ` : ''}

            <details class="details-panel">
                <summary>Raw detection data</summary>
                <div class="details-grid">
                    ${renderDetailsBlock("Unified Verdict", {
                        verdict: verdict,
                        confidence: confidence.toFixed(4),
                        adversarial_score: advScore.toFixed(4),
                        genai_score: genaiScore.toFixed(4),
                        is_duplicate: isDuplicate,
                    })}
                    ${details.lid_score != null ? renderDetailsBlock("Manifold Analysis", {
                        lid_score: details.lid_score,
                        mahalanobis_distance: details.mahalanobis_distance,
                    }) : ''}
                    ${forensic && Object.keys(forensic).length ? renderDetailsBlock("Forensic Scores", forensic) : ''}
                </div>
            </details>
        `;

        resultsStack.appendChild(card);
    });
}

// ---- Helpers ----

function renderDetectorCard(label, detector) {
    if (!detector) return '';
    const scorePct = (detector.score * 100).toFixed(1);
    return `
        <article class="metric-panel">
            <span class="metric-panel-label">${escapeHtml(label)}</span>
            <h3>${escapeHtml(detector.name)}</h3>
            <p>${escapeHtml(detector.description)}</p>
            <div class="mini-score-row">
                <span>${scorePct}%</span>
                <span>Weight ${Math.round(detector.weight * 100)}%</span>
            </div>
        </article>
    `;
}

function renderVisualCard(label, imageSrc) {
    if (!imageSrc) {
        return `
            <article class="visual-card">
                <h3>${escapeHtml(label)}</h3>
                <div class="visual-placeholder">No visualization available</div>
            </article>
        `;
    }

    return `
        <article class="visual-card">
            <h3>${escapeHtml(label)}</h3>
            <img src="${imageSrc}" alt="${escapeHtml(label)}">
        </article>
    `;
}

function renderDetailsBlock(title, details) {
    const rows = Object.entries(details || {})
        .map(([key, value]) => {
            const label = key.replace(/_/g, " ");
            return `
                <tr>
                    <td>${escapeHtml(toTitleCase(label))}</td>
                    <td>${escapeHtml(String(value))}</td>
                </tr>
            `;
        })
        .join("");

    return `
        <section class="details-block">
            <h4>${escapeHtml(title)}</h4>
            <table>
                <tbody>${rows}</tbody>
            </table>
        </section>
    `;
}

function formatVerdict(verdict) {
    const map = {
        CLEAN: "Clean",
        AI_GENERATED: "AI Generated",
        ADVERSARIAL_ATTACK: "Adversarial Attack",
        TAMPERED: "Tampered",
        DUPLICATE_FRAUD: "Duplicate Fraud",
        HYBRID_THREAT: "Hybrid Threat",
    };
    return map[verdict] || verdict;
}

function getVerdictClass(verdict) {
    const map = {
        CLEAN: "is-clean",
        AI_GENERATED: "is-ai",
        ADVERSARIAL_ATTACK: "is-adversarial",
        TAMPERED: "is-tampered",
        DUPLICATE_FRAUD: "is-duplicate",
        HYBRID_THREAT: "is-hybrid",
    };
    return map[verdict] || "is-clean";
}

// ---- Loading / Toast ----

function showLoading(message) {
    let overlay = document.getElementById("loadingOverlay");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = "loadingOverlay";
        overlay.className = "loading-overlay";
        overlay.innerHTML = `
            <div class="loading-card">
                <div class="spinner"></div>
                <p id="loadingMessage">${escapeHtml(message)}</p>
                <div class="loading-progress-track">
                    <div class="loading-progress-fill" id="loadingProgressFill"></div>
                </div>
                <span class="loading-progress-label" id="loadingProgressLabel">Preparing files</span>
            </div>
        `;
        document.body.appendChild(overlay);
    } else {
        overlay.removeAttribute("hidden");
    }

    document.getElementById("loadingMessage").textContent = message;
    startLoadingProgress();
}

function hideLoading() {
    const overlay = document.getElementById("loadingOverlay");
    if (overlay) {
        stopLoadingProgress();
        overlay.setAttribute("hidden", "hidden");
    }
}

function startLoadingProgress() {
    stopLoadingProgress();

    const fill = document.getElementById("loadingProgressFill");
    const label = document.getElementById("loadingProgressLabel");
    if (!fill || !label) return;

    const stages = [
        { progress: 12, label: "Uploading images" },
        { progress: 30, label: "Extracting EfficientNet embeddings" },
        { progress: 50, label: "Running LID + Mahalanobis analysis" },
        { progress: 68, label: "Running forensic detectors (FFT, ELA)" },
        { progress: 82, label: "Checking duplicate submissions" },
        { progress: 92, label: "Fusing detection signals" },
    ];
    let index = 0;

    fill.style.width = "5%";
    label.textContent = stages[0].label;

    loadingIntervalId = window.setInterval(() => {
        if (index < stages.length) {
            fill.style.width = `${stages[index].progress}%`;
            label.textContent = stages[index].label;
            index += 1;
            return;
        }
        fill.style.width = "95%";
        label.textContent = "Finalizing verdict";
    }, 600);
}

function stopLoadingProgress() {
    if (loadingIntervalId !== null) {
        window.clearInterval(loadingIntervalId);
        loadingIntervalId = null;
    }

    const fill = document.getElementById("loadingProgressFill");
    const label = document.getElementById("loadingProgressLabel");
    if (fill) fill.style.width = "100%";
    if (label) label.textContent = "Analysis complete";

    window.setTimeout(() => {
        const liveFill = document.getElementById("loadingProgressFill");
        if (liveFill) liveFill.style.width = "0%";
    }, 160);
}

function setAnalyzeButtonState(isBusy) {
    const button = document.getElementById("btnAnalyzeBatch");
    if (!button) return;
    button.disabled = isBusy;
    button.textContent = isBusy ? "Analyzing..." : "Analyze Images";
}

function showNotification(message, type) {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    requestAnimationFrame(() => toast.classList.add("visible"));
    setTimeout(() => {
        toast.classList.remove("visible");
        setTimeout(() => toast.remove(), 250);
    }, 2400);
}

function formatFileSize(size) {
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function toTitleCase(value) {
    return value.replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}
