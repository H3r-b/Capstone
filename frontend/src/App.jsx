import React, { useRef, useState } from "react";
import axios from "axios";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { Chart, registerables } from "chart.js";
Chart.register(...registerables);

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isCameraCapture, setIsCameraCapture] = useState(false);
  const chartRef = useRef(null);
  const resultsContainerRef = useRef(null);

  // handle file input
  function handleFileInputChange(e) {
    const f = e.target.files?.[0];
    if (f) handleSelectedFile(f, false);
  }

  function handleSelectedFile(f, fromCamera = false) {
    if (!f.type.startsWith("image/")) return alert("Please select an image file.");
    if (f.size > 10 * 1024 * 1024) return alert("File must be < 10MB");
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setIsCameraCapture(fromCamera);
  }

  function handleDrop(e) {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) handleSelectedFile(f, false);
  }

  // take photo from webcam
  async function takePhoto() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement("video");
      video.srcObject = stream;
      video.play();

      const modal = document.createElement("div");
      modal.style.position = "fixed";
      modal.style.inset = "0";
      modal.style.display = "flex";
      modal.style.alignItems = "center";
      modal.style.justifyContent = "center";
      modal.style.background = "rgba(0,0,0,0.6)";

      const holder = document.createElement("div");
      holder.style.background = "#fff";
      holder.style.padding = "12px";
      holder.style.borderRadius = "8px";
      holder.style.maxWidth = "90%";
      holder.appendChild(video);

      const capBtn = document.createElement("button");
      capBtn.textContent = "Capture";
      capBtn.style.marginTop = "8px";
      capBtn.className = "btn btn-primary";
      holder.appendChild(capBtn);

      modal.appendChild(holder);
      document.body.appendChild(modal);

      capBtn.onclick = () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);
        canvas.toBlob(
          (blob) => {
            const capturedFile = new File([blob], "capture.jpg", { type: "image/jpeg" });
            handleSelectedFile(capturedFile, true);
          },
          "image/jpeg",
          0.85
        );
        stream.getTracks().forEach((t) => t.stop());
        modal.remove();
      };
    } catch (err) {
      alert("Camera not available or permission denied.");
    }
  }

  // analyze image
  async function analyze() {
    if (!file) return alert("Choose or capture an image first");
    setLoading(true);
    setResult(null);

    const fd = new FormData();
    fd.append("file", file);

    const endpoint = isCameraCapture
      ? "http://127.0.0.1:8000/image" // ✅ camera capture
      : "http://127.0.0.1:8000/predict";   // ✅ uploaded file

    try {
      const res = await axios.post(endpoint, fd, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60000,
      });
      setResult(res.data);

      setTimeout(() => {
        const el = resultsContainerRef.current;
        if (el) {
          const cards = el.querySelectorAll(".result-card");
          cards.forEach((c, i) => setTimeout(() => c.classList.add("show"), i * 120));
        }
      }, 150);
    } catch (err) {
      console.error(err);
      alert("Backend error — check server or console.");
    } finally {
      setLoading(false);
    }
  }

  // download PDF report
  async function downloadReport() {
    if (!resultsContainerRef.current) return;
    const canvas = await html2canvas(resultsContainerRef.current, { scale: 2 });
    const img = canvas.toDataURL("image/png");
    const pdf = new jsPDF({ orientation: "portrait" });
    pdf.text("Malnutrition Report", 12, 18);
    pdf.addImage(img, "PNG", 10, 30, 190, (canvas.height / canvas.width) * 190);
    pdf.save("malnutrition_report.pdf");
  }

  // ✅ severity color + glow intensity
  function severityStyle(sev) {
    if (!sev)
      return { background: "#dcfce7", color: "#166534", borderColor: "#22c55e" };

    const s = sev.toLowerCase();

    if (s.includes("severe") || s.includes("malnourished")) {
      return {
        background: "#fee2e2",
        color: "#7f1d1d",
        borderColor: "#ef4444",
        boxShadow: "0 0 12px rgba(239,68,68,0.4)",
        animation: "pulseRed 2s infinite",
      };
    }

    if (s.includes("moder")) {
      return {
        background: "#fef3c7",
        color: "#78350f",
        borderColor: "#f59e0b",
        boxShadow: "0 0 12px rgba(245,158,11,0.4)",
      };
    }

    if (s.includes("mild")) {
      return {
        background: "#fef9c3",
        color: "#713f12",
        borderColor: "#eab308",
        boxShadow: "0 0 10px rgba(234,179,8,0.25)",
      };
    }

    if (s.includes("normal") || s.includes("healthy")) {
      return {
        background: "#dcfce7",
        color: "#166534",
        borderColor: "#22c55e",
        boxShadow: "0 0 10px rgba(34,197,94,0.25)",
      };
    }

    return { background: "#dcfce7", color: "#166534", borderColor: "#22c55e" };
  }

  // inject keyframes for pulsing severe box
  React.useEffect(() => {
    const style = document.createElement("style");
    style.innerHTML = `
      @keyframes pulseRed {
        0% { box-shadow: 0 0 10px rgba(239,68,68,0.4); }
        50% { box-shadow: 0 0 25px rgba(239,68,68,0.8); }
        100% { box-shadow: 0 0 10px rgba(239,68,68,0.4); }
      }
    `;
    document.head.appendChild(style);
  }, []);

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="h-title">Malnutrition Detection System</h1>
        <p className="h-sub">
          Advanced computer vision technology to assess nutritional status from photographs
        </p>
      </header>

      <div className="grid">
        {/* Upload Section */}
        <div className="card">
          <h3 style={{ margin: "0 0 12px 0", fontSize: 18, fontWeight: 700 }}>Upload Image</h3>

          <div
            className={"upload-area" + (preview ? "" : "")}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => document.getElementById("fileInput")?.click()}
            role="button"
          >
            {!preview ? (
              <>
                <svg
                  className="upload-ico"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#94a3b8"
                  strokeWidth="1.8"
                >
                  <path
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path d="M15 13l-3-3-3 3" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M12 16V4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <div className="upload-title">Drop your image here or click to browse</div>
                <div className="upload-sub">Supports JPG, PNG, WebP (Max 10MB)</div>
              </>
            ) : (
              <>
                <img src={preview} alt="preview" className="preview-img" />
                <div style={{ marginTop: 10, color: "var(--muted)" }}>{file?.name}</div>
                {isCameraCapture && (
                  <div
                    style={{
                      marginTop: 6,
                      color: "#0f766e",
                      fontSize: 13,
                      fontWeight: 500,
                    }}
                  >
                    📸 Captured via Camera
                  </div>
                )}
              </>
            )}
            <input
              id="fileInput"
              type="file"
              accept="image/*"
              onChange={handleFileInputChange}
              style={{ display: "none" }}
            />
          </div>

          <div style={{ marginTop: 14, display: "flex", gap: 8 }}>
            <button type="button" className="btn btn-ghost" onClick={takePhoto}>
              Take Photo
            </button>
            <div style={{ flex: 1 }} />
            <button
              onClick={analyze}
              className={`btn ${loading || !file ? "btn-disabled" : "btn-primary"}`}
              disabled={!file || loading}
            >
              {loading ? "Analyzing..." : "Analyze Image"}
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="card" ref={resultsContainerRef}>
          <h3 style={{ margin: "0 0 12px 0", fontSize: 18, fontWeight: 700 }}>Analysis Results</h3>

          {!result ? (
            <div className="result-empty">
              <svg width="84" height="84" fill="none" viewBox="0 0 24 24" style={{ opacity: 0.12 }}>
                <path
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6"
                  stroke="#111827"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M9 19V9a2 2 0 012-2h2"
                  stroke="#111827"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <div style={{ marginTop: 10, color: "var(--muted)" }}>
                Upload or capture an image and click analyze to see results
              </div>
            </div>
          ) : (
            <div className="result-card">
              <div
                style={{
                  ...severityStyle(result.severity_level),
                  border: "2px solid",
                  borderRadius: 8,
                  padding: 14,
                  transition: "all 0.4s ease",
                }}
              >
                <div style={{ fontWeight: 700, fontSize: 16 }}>
                  Severity: {result.severity_level}
                </div>
                <div style={{ color: "inherit", fontSize: 13 }}>
                  Score: {result.severity_score ? result.severity_score.toFixed(2) : "N/A"}
                </div>
                <div style={{ color: "inherit", fontSize: 13 }}>
                  Face detected: {result.face_detected ? "Yes" : "No"}
                </div>
              </div>

              <div
                style={{
                  background: "#fff9ed",
                  borderRadius: 8,
                  padding: 12,
                  border: "1px solid #fce9b8",
                  marginTop: 12,
                }}
              >
                <strong>Recommendations</strong>
                <ul style={{ marginTop: 8, paddingLeft: 18, color: "var(--muted)" }}>
                  <li>• Consult healthcare for medical diagnosis</li>
                  <li>• Monitor dietary intake and weight</li>
                </ul>
              </div>

              <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
                <button onClick={downloadReport} className="btn btn-primary" style={{ flex: 1 }}>
                  Download Report
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="footer">© {new Date().getFullYear()} Capstone MalnutriDetect</div>
    </div>
  );
}
