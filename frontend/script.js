/* ═══════════════════════════════════════════
   THE-LAG — Frontend Logic v2
   ═══════════════════════════════════════════ */

// ═══════════════════════════════════════════════
// IMPORTANT: Replace with your Render backend URL
// e.g. "https://thelag-api.onrender.com"
// For local development use "http://localhost:8000"
// ═══════════════════════════════════════════════
const API = window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://YOUR-RENDER-APP-NAME.onrender.com";
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ──────────── Language Switcher ────────────

let currentLang = "en";

$$(".lang-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        currentLang = btn.dataset.lang;
        $$(".lang-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        applyLang();
    });
});

function applyLang() {
    $$("[data-en][data-gr]").forEach((el) => {
        const text = el.getAttribute(`data-${currentLang}`);
        if (text) {
            // Preserve innerHTML for elements with <strong> etc.
            if (text.includes("<")) {
                el.innerHTML = text;
            } else {
                el.textContent = text;
            }
        }
    });
    document.documentElement.lang = currentLang === "gr" ? "el" : "en";
    // Update toggle buttons if they exist
    if (typeof updateToggleTexts === "function") updateToggleTexts();
}

// ──────────── Active Nav Link on Scroll ────────────

const sections = $$("section[id]");
const navLinks = $$(".nav-link");

function updateActiveNav() {
    const scrollY = window.scrollY + 100;
    sections.forEach((section) => {
        const top = section.offsetTop;
        const height = section.offsetHeight;
        const id = section.getAttribute("id");
        if (scrollY >= top && scrollY < top + height) {
            navLinks.forEach((link) => {
                link.classList.remove("active");
                if (link.getAttribute("href") === `#${id}`) {
                    link.classList.add("active");
                }
            });
        }
    });
}

window.addEventListener("scroll", updateActiveNav, { passive: true });

// ──────────── Mobile Menu ────────────

const hamburger = $("#nav-hamburger");
const mobileMenu = $("#mobile-menu");

hamburger.addEventListener("click", () => {
    hamburger.classList.toggle("open");
    mobileMenu.classList.toggle("open");
});

$$(".mobile-link").forEach((link) => {
    link.addEventListener("click", () => {
        hamburger.classList.remove("open");
        mobileMenu.classList.remove("open");
    });
});

// ──────────── Expandable About Cards ────────────

$$(".about-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
        const card = btn.closest(".about-card");
        const isExpanded = card.classList.toggle("expanded");

        // Update button text based on language
        if (isExpanded) {
            btn.textContent = btn.getAttribute(`data-${currentLang}-hide`);
        } else {
            btn.textContent = btn.getAttribute(`data-${currentLang}-show`);
        }
    });
});

// Update toggle button text on language change
function updateToggleTexts() {
    $$(".about-toggle").forEach((btn) => {
        const card = btn.closest(".about-card");
        const isExpanded = card.classList.contains("expanded");
        if (isExpanded) {
            btn.textContent = btn.getAttribute(`data-${currentLang}-hide`);
        } else {
            btn.textContent = btn.getAttribute(`data-${currentLang}-show`);
        }
    });
}

// ──────────── File Upload ────────────

const uploadZone = $("#upload-zone");
const fileInput = $("#file-input");
const fileInfo = $("#file-info");
const fileName = $("#file-name");
const btnClearFile = $("#btn-clear-file");
const btnRun = $("#btn-run");
const progressSec = $("#progress-section");
const progressBar = $("#progress-bar");
const progressText = $("#progress-text");
const resultsSec = $("#results");
const shapImg = $("#shap-img");
const shapPlaceholder = $("#shap-placeholder");
const xcorrImg = $("#xcorr-img");
const xcorrPlaceholder = $("#xcorr-placeholder");

let selectedFile = null;

uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["xlsx", "csv"].includes(ext)) return;
    selectedFile = file;
    fileName.textContent = file.name;
    uploadZone.classList.add("hidden");
    fileInfo.classList.remove("hidden");
    btnRun.disabled = false;
}

btnClearFile.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    fileInfo.classList.add("hidden");
    uploadZone.classList.remove("hidden");
    btnRun.disabled = true;
});

// ──────────── Run Pipeline ────────────

btnRun.addEventListener("click", runPipeline);

async function runPipeline() {
    if (!selectedFile) return;

    // Clear previous results
    clearResults();

    btnRun.disabled = true;
    $(".btn-label").textContent = currentLang === "gr" ? "Εκτέλεση…" : "Running…";
    $(".btn-spinner").classList.remove("hidden");
    progressSec.classList.remove("hidden");
    progressBar.style.width = "0%";
    progressText.textContent = currentLang === "gr" ? "Ανέβασμα αρχείου…" : "Uploading file…";

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const res = await fetch(`${API}/upload-and-run`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        const contentType = res.headers.get("content-type") || "";

        if (contentType.includes("text/event-stream")) {
            await handleSSE(res);
        } else {
            const data = await res.json();
            if (data.status === "ok") {
                progressBar.style.width = "100%";
                progressText.textContent = currentLang === "gr" ? "Ολοκληρώθηκε." : "Pipeline complete.";
                await loadResults();
            } else {
                throw new Error(data.message || "Pipeline failed.");
            }
        }
    } catch (err) {
        progressText.textContent = `Error: ${err.message}`;
        progressBar.style.width = "0%";
    } finally {
        $(".btn-label").textContent = currentLang === "gr" ? "Εκτέλεση Pipeline" : "Run Pipeline";
        $(".btn-spinner").classList.add("hidden");
        btnRun.disabled = false;
    }
}

function clearResults() {
    // Hide results section
    resultsSec.classList.add("hidden");

    // Clear metric values
    $("#xgb-acc").textContent = "—";
    $("#xgb-f1").textContent = "—";
    $("#mlp-acc").textContent = "—";
    $("#mlp-f1").textContent = "—";

    // Clear reports
    $("#xgb-report").textContent = "";
    $("#mlp-report").textContent = "";

    // Clear confusion matrices
    $("#xgb-cm").innerHTML = "";
    $("#mlp-cm").innerHTML = "";

    // Reset SHAP image
    shapImg.src = "";
    shapImg.classList.add("hidden");
    shapPlaceholder.classList.remove("hidden");

    // Reset cross-correlation image
    xcorrImg.src = "";
    xcorrImg.classList.add("hidden");
    xcorrPlaceholder.classList.remove("hidden");

    // Reset progress
    progressBar.style.width = "0%";
    progressText.textContent = "";
}

async function handleSSE(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const stages = {
        preprocessing: 20,
        training: 50,
        evaluation: 75,
        shap: 90,
        cross_correlation: 95,
        done: 100,
    };

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
            if (line.startsWith("data: ")) {
                try {
                    const evt = JSON.parse(line.slice(6));
                    if (evt.stage && stages[evt.stage] !== undefined) {
                        progressBar.style.width = stages[evt.stage] + "%";
                    }
                    if (evt.message) {
                        progressText.textContent = evt.message;
                    }
                    if (evt.stage === "done") {
                        await loadResults();
                    }
                    if (evt.error) {
                        progressText.textContent = `Error: ${evt.error}`;
                    }
                } catch {
                    // ignore
                }
            }
        }
    }
}

// ──────────── Load Results ────────────

async function loadResults() {
    try {
        const metricsRes = await fetch(`${API}/metrics`);
        if (metricsRes.ok) {
            const m = await metricsRes.json();
            renderMetrics(m);
        }

        const shapRes = await fetch(`${API}/shap-summary`);
        if (shapRes.ok) {
            const blob = await shapRes.blob();
            shapImg.src = URL.createObjectURL(blob);
            shapImg.classList.remove("hidden");
            shapPlaceholder.classList.add("hidden");
        }

        try {
            const xcorrRes = await fetch(`${API}/cross-correlation`);
            if (xcorrRes.ok) {
                const blob = await xcorrRes.blob();
                xcorrImg.src = URL.createObjectURL(blob);
                xcorrImg.classList.remove("hidden");
                xcorrPlaceholder.classList.add("hidden");
            }
        } catch { /* optional */ }

        resultsSec.classList.remove("hidden");
    } catch (err) {
        console.error("Failed to load results:", err);
    }
}

function renderMetrics(m) {
    $("#xgb-acc").textContent = fmt(m.xgb_accuracy ?? m.xgboost?.accuracy);
    $("#xgb-f1").textContent = fmt(m.xgb_f1 ?? m.xgboost?.f1);
    $("#mlp-acc").textContent = fmt(m.mlp_accuracy ?? m.mlp?.accuracy);
    $("#mlp-f1").textContent = fmt(m.mlp_f1 ?? m.mlp?.f1);

    const xgbReport = m.xgb_report ?? m.xgboost?.classification_report ?? "";
    const mlpReport = m.mlp_report ?? m.mlp?.classification_report ?? "";
    $("#xgb-report").textContent = typeof xgbReport === "string" ? xgbReport : JSON.stringify(xgbReport, null, 2);
    $("#mlp-report").textContent = typeof mlpReport === "string" ? mlpReport : JSON.stringify(mlpReport, null, 2);

    const xgbCM = m.xgb_cm ?? m.xgboost?.confusion_matrix;
    const mlpCM = m.mlp_cm ?? m.mlp?.confusion_matrix;
    if (xgbCM) renderCM($("#xgb-cm"), xgbCM);
    if (mlpCM) renderCM($("#mlp-cm"), mlpCM);
}

function renderCM(container, cm) {
    if (!Array.isArray(cm) || cm.length === 0) {
        container.textContent = "No data";
        return;
    }
    const labels = cm.length === 2 ? ["Negative (0)", "Positive (1)"] : cm.map((_, i) => `Class ${i}`);
    let html = "<table><thead><tr><th></th>";
    labels.forEach((l) => (html += `<th>Pred ${l}</th>`));
    html += "</tr></thead><tbody>";
    cm.forEach((row, i) => {
        html += `<tr><th>Actual ${labels[i]}</th>`;
        row.forEach((val, j) => {
            const cls = i === j ? "cm-diag" : "cm-off";
            html += `<td class="${cls}">${val}</td>`;
        });
        html += "</tr>";
    });
    html += "</tbody></table>";
    container.innerHTML = html;
}

function fmt(val) {
    if (val === undefined || val === null) return "—";
    const n = parseFloat(val);
    if (isNaN(n)) return val;
    return (n * (n <= 1 ? 100 : 1)).toFixed(1) + "%";
}

// ──────────── Scroll Reveal (Intersection Observer) ────────────

const revealObserver = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
                revealObserver.unobserve(entry.target); // animate once
            }
        });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
);

$$(".reveal").forEach((el) => revealObserver.observe(el));

// ──────────── Back to Top + Navbar Shrink ────────────

const backToTop = $("#back-to-top");
const navbar = $(".navbar");

window.addEventListener("scroll", () => {
    const y = window.scrollY;

    // Back to top
    if (y > 500) {
        backToTop.classList.add("show");
    } else {
        backToTop.classList.remove("show");
    }

    // Navbar shrink
    if (y > 50) {
        navbar.classList.add("scrolled");
    } else {
        navbar.classList.remove("scrolled");
    }
}, { passive: true });

backToTop.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
});

// ──────────── Smooth Nav Transitions ────────────

$$(".nav-link, .mobile-link, .hero-btn").forEach((link) => {
    link.addEventListener("click", (e) => {
        const href = link.getAttribute("href");
        if (!href || !href.startsWith("#")) return;
        e.preventDefault();
        const target = document.querySelector(href);
        if (!target) return;

        // Update active state immediately
        $$(".nav-link").forEach((l) => l.classList.remove("active"));
        const matchingNav = $(`.nav-link[href="${href}"]`);
        if (matchingNav) matchingNav.classList.add("active");

        // Smooth scroll to target
        const navHeight = navbar.offsetHeight;
        const top = target.offsetTop - navHeight - 12;
        window.scrollTo({ top, behavior: "smooth" });
    });
});

// ──────────── Download Results ────────────

const btnDownload = $("#btn-download");
if (btnDownload) {
    btnDownload.addEventListener("click", downloadResults);
}

async function downloadResults() {
    btnDownload.disabled = true;
    const originalText = btnDownload.querySelector("span").textContent;
    btnDownload.querySelector("span").textContent = currentLang === "gr" ? "Λήψη…" : "Downloading…";

    try {
        // Fetch metrics JSON
        let metricsBlob = null;
        try {
            const res = await fetch(`${API}/metrics`);
            if (res.ok) {
                const data = await res.json();
                metricsBlob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            }
        } catch { /* skip */ }

        // Fetch SHAP image
        let shapBlob = null;
        try {
            const res = await fetch(`${API}/shap-summary`);
            if (res.ok) shapBlob = await res.blob();
        } catch { /* skip */ }

        // Fetch cross-correlation image
        let xcorrBlob = null;
        try {
            const res = await fetch(`${API}/cross-correlation`);
            if (res.ok) xcorrBlob = await res.blob();
        } catch { /* skip */ }

        // Download each file
        if (metricsBlob) triggerDownload(metricsBlob, "metrics.json");
        if (shapBlob) triggerDownload(shapBlob, "shap_summary.png");
        if (xcorrBlob) triggerDownload(xcorrBlob, "cross_correlation.png");

        if (!metricsBlob && !shapBlob && !xcorrBlob) {
            alert(currentLang === "gr" ? "Δεν βρέθηκαν αποτελέσματα." : "No results found.");
        }
    } catch (err) {
        console.error("Download error:", err);
    } finally {
        btnDownload.disabled = false;
        btnDownload.querySelector("span").textContent = originalText;
    }
}

function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}