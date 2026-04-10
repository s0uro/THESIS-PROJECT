"""
THE-LAG — FastAPI Backend
SSE-streamed ML pipeline: upload → preprocess → train → evaluate → SHAP → cross-correlation.
No authentication. CORS enabled for frontend.
"""

import os
import json
import asyncio
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from config import (
    DATA_DIR, METRICS_PATH,
    SHAP_SUMMARY_PATH, SHAP_DEPENDENCE_PATH, XCORR_PLOT_PATH,
)

app = FastAPI(title="THE-LAG API", version="2.0")

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_event(stage: str, message: str, error: str = None) -> str:
    """Format a Server-Sent Event."""
    data = {"stage": stage, "message": message}
    if error:
        data["error"] = error
    return f"data: {json.dumps(data)}\n\n"


async def run_pipeline_sse(filepath: str):
    """
    Generator that runs the full ML pipeline and yields SSE events.
    Each stage is imported and run inline (not subprocess) for simplicity.
    """
    try:
        # ── Stage 1: Preprocessing ──
        yield sse_event("preprocessing", "Preprocessing: cleaning data, engineering features...")
        await asyncio.sleep(0)

        from preprocessing import run_preprocessing
        run_preprocessing(filepath)

        yield sse_event("preprocessing_done", "Preprocessing complete.")

        # ── Stage 2: Training ──
        yield sse_event("training", "Training: XGBoost (300 trees) + MLP (128-64-32)...")
        await asyncio.sleep(0)

        from training import run_training
        run_training()

        yield sse_event("training_done", "Training complete.")

        # ── Stage 3: Evaluation ──
        yield sse_event("evaluation", "Evaluation: computing accuracy, F1, confusion matrices...")
        await asyncio.sleep(0)

        from evaluation import run_evaluation
        run_evaluation()

        yield sse_event("evaluation_done", "Evaluation complete.")

        # ── Stage 4: SHAP ──
        yield sse_event("shap", "SHAP: computing feature importance (TreeExplainer)...")
        await asyncio.sleep(0)

        from shap_explainability import run_shap
        run_shap()

        yield sse_event("shap_done", "SHAP explainability complete.")

        # ── Stage 5: Cross-Correlation ──
        yield sse_event("cross_correlation", "Cross-correlation: analyzing ET vs FT alignment...")
        await asyncio.sleep(0)

        from cross_correlation import run_cross_correlation
        run_cross_correlation()

        yield sse_event("cross_correlation_done", "Cross-correlation complete.")

        # ── Done ──
        yield sse_event("done", "Pipeline finished successfully!")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[api] Pipeline error: {error_msg}")
        print(traceback.format_exc())
        yield sse_event("error", "Pipeline failed.", error=error_msg)


# ═══════════════ ENDPOINTS ═══════════════

@app.post("/upload-and-run")
async def upload_and_run(file: UploadFile = File(...)):
    """
    Upload dataset and run full pipeline with SSE streaming progress.
    """
    # Validate file extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("xlsx", "csv"):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Only .xlsx and .csv files are supported."},
        )

    # Save uploaded file
    save_path = os.path.join(DATA_DIR, f"uploaded_data.{ext}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    print(f"[api] File saved: {save_path} ({len(content)} bytes)")

    # Return SSE stream
    return StreamingResponse(
        run_pipeline_sse(save_path),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/metrics")
async def get_metrics():
    """Return evaluation metrics JSON."""
    if not os.path.exists(METRICS_PATH):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No metrics found. Run the pipeline first."},
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return JSONResponse(content=metrics)


@app.get("/shap-summary")
async def get_shap_summary():
    """Return SHAP summary plot as PNG."""
    if not os.path.exists(SHAP_SUMMARY_PATH):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No SHAP plot found. Run the pipeline first."},
        )
    return FileResponse(SHAP_SUMMARY_PATH, media_type="image/png")


@app.get("/shap-dependence")
async def get_shap_dependence():
    """Return SHAP dependence plot as PNG."""
    if not os.path.exists(SHAP_DEPENDENCE_PATH):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No SHAP dependence plot found."},
        )
    return FileResponse(SHAP_DEPENDENCE_PATH, media_type="image/png")


@app.get("/cross-correlation")
async def get_cross_correlation():
    """Return cross-correlation plot as PNG."""
    if not os.path.exists(XCORR_PLOT_PATH):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No cross-correlation plot found."},
        )
    return FileResponse(XCORR_PLOT_PATH, media_type="image/png")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "THE-LAG API"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)