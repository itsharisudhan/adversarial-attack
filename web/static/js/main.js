const MAX_BATCH_IMAGES = window.__AI_APP_CONFIG__.maxBatchImages;

let selectedFiles = [];
let loadingIntervalId = null;
let historyEnabled = false;

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
        historyEnabled = Boolean(data.history_enabled);
        document.getElementById("appStatus").textContent = `Ready · ${data.max_batch_images} image max`;
        setupHistoryState(data);
        if (historyEnabled) {
            loadHistory();
        }
    } catch (error) {
        document.getElementById("appStatus").textContent = "Stats unavailable";
    }
}

function setupHistoryState(stats) {
    const panel = document.getElementById("historyPanel");
    const backend = document.getElementById("historyBackend");
    const hint = document.getElementById("historyHint");

    if (!panel || !backend || !hint) {
        return;
    }

    if (!historyEnabled) {
        panel.hidden = true;
        return;
    }

    backend.textContent = stats.history_backend || "Configured";
    hint.textContent = "Each saved card shows the uploaded image plus FFT and ELA analysis outputs.";
    panel.hidden = false;
}

async function loadHistory() {
    if (!historyEnabled) {
        return;
    }

    try {
        const response = await fetch("/api/history");
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Could not load history");
        }
        renderHistory(data.results || []);
    } catch (error) {
        showNotification(error.message, "error");
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
    showLoading(`Analyzing ${selectedFiles.length} image${selectedFiles.length === 1 ? "" : "s"}...`);

    try {
        const formData = new FormData();
        selectedFiles.forEach((file) => formData.append("images", file));

        const response = await fetch("/api/detect_ai_batch", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Batch analysis failed");
        }

        renderSummary(data.results);
        renderResults(data.results);
        if (historyEnabled) {
            loadHistory();
        }
        showNotification("Analysis complete.", "success");
    } catch (error) {
        showNotification(error.message, "error");
    } finally {
        hideLoading();
        setAnalyzeButtonState(false);
    }
}

function renderSummary(results) {
    const summaryPanel = document.getElementById("summaryPanel");
    const summaryGrid = document.getElementById("summaryGrid");

    let aiCount = 0;
    let uncertainCount = 0;
    let realCount = 0;

    results.forEach((result) => {
        if (result.ensemble_score >= 0.40) {
            aiCount += 1;
        } else if (result.ensemble_score >= 0.25) {
            uncertainCount += 1;
        } else {
            realCount += 1;
        }
    });

    summaryGrid.innerHTML = `
        <div class="summary-card">
            <span class="summary-label">Images analyzed</span>
            <strong>${results.length}</strong>
        </div>
        <div class="summary-card">
            <span class="summary-label">AI / likely AI</span>
            <strong>${aiCount}</strong>
        </div>
        <div class="summary-card">
            <span class="summary-label">Uncertain</span>
            <strong>${uncertainCount}</strong>
        </div>
        <div class="summary-card">
            <span class="summary-label">Likely real / real</span>
            <strong>${realCount}</strong>
        </div>
    `;

    summaryPanel.hidden = false;
}

function renderResults(results) {
    const resultsStack = document.getElementById("resultsStack");
    resultsStack.innerHTML = "";

    results.forEach((result, index) => {
        const card = document.createElement("section");
        card.className = "panel result-card";
        card.style.animationDelay = `${index * 0.08}s`;

        const verdictClass = getVerdictClass(result.ensemble_score);
        const scorePct = (result.ensemble_score * 100).toFixed(1);

        card.innerHTML = `
            <div class="result-head">
                <div>
                    <p class="section-label">Result ${index + 1}</p>
                    <h2>${escapeHtml(result.filename)}</h2>
                </div>
                <div class="verdict-chip ${verdictClass}">${escapeHtml(result.verdict)}</div>
            </div>

            <div class="verdict-bar-wrap">
                <div class="verdict-bar-label">
                    <span>AI probability</span>
                    <strong>${scorePct}%</strong>
                </div>
                <div class="verdict-bar">
                    <div class="verdict-bar-fill ${verdictClass}" style="width:${scorePct}%"></div>
                </div>
            </div>

            <div class="metric-grid">
                ${renderDetectorCard("FFT", result.detectors.fft)}
                ${renderDetectorCard("ELA", result.detectors.ela)}
                ${renderDetectorCard("Stats", result.detectors.statistics)}
                ${renderDetectorCard("Texture", result.detectors.texture)}
            </div>

            <div class="visual-grid">
                ${renderVisualCard("FFT Spectrum", result.visualizations.fft_spectrum)}
                ${renderVisualCard("ELA Heatmap", result.visualizations.ela_heatmap)}
            </div>

            <details class="details-panel">
                <summary>Detailed metrics</summary>
                <div class="details-grid">
                    ${renderDetailsBlock("Image", {
                        original_size: `${result.image_info.original_size[0]}x${result.image_info.original_size[1]}`,
                        analyzed_size: `${result.image_info.analyzed_size[0]}x${result.image_info.analyzed_size[1]}`,
                    })}
                    ${renderDetailsBlock(result.detectors.fft.name, result.detectors.fft.details)}
                    ${renderDetailsBlock(result.detectors.ela.name, result.detectors.ela.details)}
                    ${renderDetailsBlock(result.detectors.statistics.name, result.detectors.statistics.details)}
                    ${renderDetailsBlock(result.detectors.texture.name, result.detectors.texture.details)}
                </div>
            </details>
        `;

        resultsStack.appendChild(card);
    });
}

function renderHistory(results) {
    const grid = document.getElementById("historyGrid");
    const panel = document.getElementById("historyPanel");

    if (!grid || !panel || !historyEnabled) {
        return;
    }

    panel.hidden = false;
    if (!results.length) {
        grid.innerHTML = `
            <article class="history-empty">
                <strong>No saved analyses yet</strong>
                <span>Run an analysis to populate the review history.</span>
            </article>
        `;
        return;
    }

    grid.innerHTML = results
        .map((result) => {
            const score = Number(result.ensemble_score || 0);
            const verdictClass = getVerdictClass(score);
            const scorePct = (score * 100).toFixed(1);
            const createdAt = result.created_at
                ? new Date(result.created_at).toLocaleString()
                : "Saved recently";

            return `
                <article class="history-card">
                    <div class="history-card-head">
                        <div>
                            <strong>${escapeHtml(result.filename || "image")}</strong>
                            <span>${escapeHtml(createdAt)}</span>
                        </div>
                        <div class="verdict-chip ${verdictClass}">${escapeHtml(result.verdict || "Saved")}</div>
                    </div>
                    <div class="history-visual-strip">
                        ${renderHistoryImage("Input", result.input_preview_url)}
                        ${renderHistoryImage("FFT", result.fft_spectrum_url)}
                        ${renderHistoryImage("ELA", result.ela_heatmap_url)}
                    </div>
                    <div class="mini-score-row">
                        <span>AI probability ${scorePct}%</span>
                        <span>${escapeHtml(result.verdict_short || "")}</span>
                    </div>
                </article>
            `;
        })
        .join("");
}

function renderHistoryImage(label, imageSrc) {
    if (!imageSrc) {
        return `
            <div class="history-thumb is-empty">
                <span>${escapeHtml(label)}</span>
            </div>
        `;
    }

    return `
        <figure class="history-thumb">
            <img src="${imageSrc}" alt="${escapeHtml(label)}">
            <figcaption>${escapeHtml(label)}</figcaption>
        </figure>
    `;
}

function renderDetectorCard(label, detector) {
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

function getVerdictClass(score) {
    if (score >= 0.55) {
        return "is-ai";
    }
    if (score >= 0.40) {
        return "is-likely-ai";
    }
    if (score >= 0.25) {
        return "is-mixed";
    }
    return "is-real";
}

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
    if (!fill || !label) {
        return;
    }

    const stages = [
        { progress: 18, label: "Preparing files" },
        { progress: 38, label: "Uploading batch" },
        { progress: 64, label: "Running forensic detectors" },
        { progress: 86, label: "Collecting scores" },
    ];
    let index = 0;

    fill.style.width = "8%";
    label.textContent = stages[0].label;

    loadingIntervalId = window.setInterval(() => {
        if (index < stages.length) {
            fill.style.width = `${stages[index].progress}%`;
            label.textContent = stages[index].label;
            index += 1;
            return;
        }

        fill.style.width = "92%";
        label.textContent = "Finalizing results";
    }, 550);
}

function stopLoadingProgress() {
    if (loadingIntervalId !== null) {
        window.clearInterval(loadingIntervalId);
        loadingIntervalId = null;
    }

    const fill = document.getElementById("loadingProgressFill");
    const label = document.getElementById("loadingProgressLabel");
    if (fill) {
        fill.style.width = "100%";
    }
    if (label) {
        label.textContent = "Analysis complete";
    }

    window.setTimeout(() => {
        const liveFill = document.getElementById("loadingProgressFill");
        if (liveFill) {
            liveFill.style.width = "0%";
        }
    }, 160);
}

function setAnalyzeButtonState(isBusy) {
    const button = document.getElementById("btnAnalyzeBatch");
    if (!button) {
        return;
    }

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
    if (size < 1024) {
        return `${size} B`;
    }
    if (size < 1024 * 1024) {
        return `${(size / 1024).toFixed(1)} KB`;
    }
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
