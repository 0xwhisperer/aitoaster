// The Fixer - frontend logic
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ---------- theme ----------
  const themeToggle = $("themeToggle");
  themeToggle.addEventListener("click", () => {
    const root = document.documentElement;
    const cur = root.getAttribute("data-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const effectiveDark = cur ? cur === "dark" : prefersDark;
    root.setAttribute("data-theme", effectiveDark ? "light" : "dark");
  });

  // ---------- tool catalog ----------
  const TOOLS = [
    { id: "trim_silence", group: "chainGroupCleanup", name: "Trim silence", desc: "Removes leading/trailing true silence at the very start and end." },
    { id: "dc_offset", group: "chainGroupCleanup", name: "DC offset correction", desc: "Centers the waveform on zero if it's biased up or down." },
    { id: "fix_transients", group: "chainGroupCleanup", name: "Surgical transient/pop fix", desc: "Auto-detects sharp pops/spikes and gently limits just that moment." },
    { id: "high_pass", group: "chainGroupCleanup", name: "High-pass filter", desc: "Removes inaudible sub-30Hz rumble that eats into headroom." },
    { id: "linear_fix", group: "chainGroupAI", name: "Linear model fix", desc: "Gradient-optimized correction targeting the fakeprint logistic-regression detector." },
    { id: "cnn_fix", group: "chainGroupAI", name: "CNN model fix", desc: "Whole-track joint optimization targeting the CQT-cepstrum CNN detector. Slower." },
    { id: "fix_phase", group: "chainGroupMaster", name: "Stereo phase correction", desc: "Corrects out-of-phase content that would cancel out in mono playback." },
    { id: "normalize_lufs", group: "chainGroupMaster", name: "LUFS loudness normalization", desc: "Targets -14 LUFS, the standard streaming-platform loudness reference." },
    { id: "multiband_compress", group: "chainGroupMaster", name: "Multiband compression", desc: "Gentle 3-band dynamics smoothing for tonal balance." },
    { id: "true_peak_limit", group: "chainGroupMaster", name: "True-peak limiter", desc: "Brick-wall safety ceiling at -1dBTP, accounting for inter-sample peaks." },
  ];

  let state = {
    fileId: null,
    filename: null,
    analysis: null,
    selected: new Set(),
    jobId: null,
    pollTimer: null,
    result: null,
    abMode: "orig", // 'orig' | 'fixed'
    audioCtx: null,
    waveData: null,
  };

  // ---------- upload ----------
  const uploadZone = $("uploadZone");
  const fileInput = $("fileInput");

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  $("newFileBtn").addEventListener("click", () => {
    resetWorkspace();
  });

  function resetWorkspace() {
    state = { ...state, fileId: null, analysis: null, selected: new Set(), jobId: null, result: null };
    $("workspace").classList.remove("active");
    $("results").classList.remove("active");
    $("progressPanel").classList.remove("active");
    fileInput.value = "";
  }

  async function handleFile(file) {
    uploadZone.classList.add("dragover");
    uploadZone.querySelector(".primary").textContent = "Uploading & decoding…";
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      state.fileId = data.file_id;
      state.filename = data.filename;
      uploadZone.querySelector(".primary").textContent = "Drop an audio file, or click to browse";
      uploadZone.classList.remove("dragover");

      $("fileName").textContent = data.filename;
      $("fileMeta").textContent = `${fmtDuration(data.duration_sec)} · ${data.samples.toLocaleString()} samples`;

      const stem = data.filename.replace(/\.[^.]+$/, "");
      $("outputNameInput").value = `${stem}_fixed.wav`;

      $("workspace").classList.add("active");
      $("results").classList.remove("active");
      renderToolChain();
      await runAnalysis();
    } catch (err) {
      uploadZone.querySelector(".primary").textContent = "Upload failed — try again";
      console.error(err);
    }
  }

  function fmtDuration(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  // ---------- analysis ----------
  async function runAnalysis() {
    const res = await fetch(`/api/analyze/${state.fileId}`);
    const data = await res.json();
    state.analysis = data;
    renderAnalysis(data);
    state.selected = new Set(data.recommended_tools);
    renderToolChain();
  }

  function scoreClass(pct, kind) {
    if (kind === "linear") {
      if (pct < 1) return "good";
      if (pct < 20) return "warn";
      return "crit";
    } else {
      if (pct < 50) return "good";
      if (pct < 80) return "warn";
      return "crit";
    }
  }

  function renderAnalysis(data) {
    const linPct = data.scores.linear_pct;
    const cnnPct = data.scores.cnn_pct;

    const linCls = scoreClass(linPct, "linear");
    const cnnCls = scoreClass(cnnPct, "cnn");

    $("linearScore").textContent = `${linPct.toFixed(linPct < 1 ? 3 : 1)}%`;
    $("linearScore").className = `value mono ${linCls}`;
    $("linearMeter").style.width = `${Math.min(100, linPct)}%`;
    $("linearMeter").className = `meter-fill ${linCls}`;
    $("linearPill").innerHTML = data.scores.passes_linear
      ? `<span class="pill good">Reads as human</span>`
      : `<span class="pill crit">Flagged as AI</span>`;

    $("cnnScore").textContent = `${cnnPct.toFixed(1)}%`;
    $("cnnScore").className = `value mono ${cnnCls}`;
    $("cnnMeter").style.width = `${Math.min(100, cnnPct)}%`;
    $("cnnMeter").className = `meter-fill ${cnnCls}`;
    $("cnnPill").innerHTML = data.scores.passes_cnn
      ? `<span class="pill good">Reads as human</span>`
      : `<span class="pill crit">Flagged as AI</span>`;

    const stats = [
      { k: "LUFS (integrated)", v: `${data.lufs.toFixed(1)} LU` },
      { k: "Stereo correlation", v: data.stereo_correlation.toFixed(2), flag: data.stereo_correlation < 0.1 },
      { k: "DC offset (L/R)", v: `${data.dc_offset.l.toFixed(5)} / ${data.dc_offset.r.toFixed(5)}` },
      { k: "Leading/trailing silence", v: `${data.silence.lead_ms || 0}ms / ${data.silence.trail_ms || 0}ms` },
      { k: "Transients detected", v: data.transients.length, flag: data.transients.length > 0 },
      { k: "Spectral tilt (low/mid/high)", v: `${data.spectral_tilt["low (20-250Hz)"].toFixed(0)} / ${data.spectral_tilt["mid (250-4000Hz)"].toFixed(0)} / ${data.spectral_tilt["high (4000-20000Hz)"].toFixed(0)} dB` },
    ];
    $("statList").innerHTML = stats.map(s => `
      <div class="stat">
        <div class="k">${s.k}</div>
        <div class="v ${s.flag ? "flag" : ""}">${s.v}</div>
      </div>`).join("");

    drawSpectrum($("spectrumCanvas"), data.spectrum.freqs, data.spectrum.psd_db);
  }

  function drawSpectrum(canvas, freqs, psdDb, overlayFreqs, overlayPsd) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim();
    const faint = styles.getPropertyValue("--text-faint").trim();
    const rackLine = styles.getPropertyValue("--rack-line").trim();

    // grid
    ctx.strokeStyle = rackLine;
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (h / 4) * i;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    const minDb = -100, maxDb = 20;
    const logMin = Math.log10(20), logMax = Math.log10(20000);

    function pathFor(freqs, psdDb) {
      const pts = [];
      for (let i = 0; i < freqs.length; i++) {
        const f = Math.max(freqs[i], 20);
        const x = ((Math.log10(f) - logMin) / (logMax - logMin)) * w;
        const db = Math.max(minDb, Math.min(maxDb, psdDb[i]));
        const y = h - ((db - minDb) / (maxDb - minDb)) * h;
        pts.push([x, y]);
      }
      return pts;
    }

    function drawFill(pts, color, alpha) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], h);
      for (const [x, y] of pts) ctx.lineTo(x, y);
      ctx.lineTo(pts[pts.length - 1][0], h);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    function drawLine(pts, color, width) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (const [x, y] of pts) ctx.lineTo(x, y);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    }

    if (overlayFreqs) {
      const beforePts = pathFor(freqs, psdDb);
      drawLine(beforePts, faint, 1.5);
      const afterPts = pathFor(overlayFreqs, overlayPsd);
      drawFill(afterPts, accent, 0.12);
      drawLine(afterPts, accent, 2);
    } else {
      const pts = pathFor(freqs, psdDb);
      drawFill(pts, accent, 0.15);
      drawLine(pts, accent, 1.75);
    }
  }

  // ---------- tool chain UI ----------
  function renderToolChain() {
    const groups = { chainGroupCleanup: [], chainGroupAI: [], chainGroupMaster: [] };
    for (const t of TOOLS) groups[t.group].push(t);

    for (const [groupId, tools] of Object.entries(groups)) {
      const rec = state.analysis ? new Set(state.analysis.recommended_tools) : new Set();
      $(groupId).innerHTML = tools.map(t => {
        const checked = state.selected.has(t.id);
        const recommended = rec.has(t.id);
        return `
        <div class="tool-row ${checked ? "checked" : ""}" data-tool="${t.id}">
          <div class="box"></div>
          <div class="info">
            <div class="name">${t.name}${recommended ? `<span class="rec-badge">Recommended</span>` : ""}</div>
            <div class="desc">${t.desc}</div>
          </div>
        </div>`;
      }).join("");
    }

    document.querySelectorAll(".tool-row").forEach(row => {
      row.addEventListener("click", () => {
        const id = row.dataset.tool;
        if (state.selected.has(id)) state.selected.delete(id);
        else state.selected.add(id);
        renderToolChain();
      });
    });

    $("runCount").textContent = `${state.selected.size} selected`;
  }

  $("selectRecBtn").addEventListener("click", () => {
    if (!state.analysis) return;
    state.selected = new Set(state.analysis.recommended_tools);
    renderToolChain();
  });

  // ---------- run pipeline ----------
  $("runBtn").addEventListener("click", async () => {
    if (!state.fileId || state.selected.size === 0) return;
    $("progressPanel").classList.add("active");
    $("results").classList.remove("active");
    $("logBox").innerHTML = "";
    $("progressFill").style.width = "6%";
    $("runBtn").disabled = true;

    const outputName = $("outputNameInput").value.trim();
    const res = await fetch(`/api/process/${state.fileId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tools: Array.from(state.selected), options: {}, output_name: outputName || undefined }),
    });
    const data = await res.json();
    if (data.error) { appendLog(data.error, true); $("runBtn").disabled = false; return; }
    state.jobId = data.job_id;
    pollJob();
  });

  function updateProgress(data) {
    const total = data.total_steps;
    const idx = data.current_step_idx;

    if (data.status === "done") {
      $("progressFill").style.width = "100%";
      $("progressStepLabel").textContent = "Complete";
      $("progressStepCount").textContent = "";
      return;
    }

    if (total == null || idx == null) {
      // no step info yet (job just started)
      $("progressFill").style.width = "4%";
      $("progressStepLabel").textContent = data.progress_msg || "Starting…";
      $("progressStepCount").textContent = "";
      return;
    }

    // base progress: fraction of steps fully completed
    let fracWithinStep = 0;
    if (data.sub_progress && data.sub_progress.total) {
      fracWithinStep = Math.min(1, data.sub_progress.current / data.sub_progress.total);
    }
    const overallFrac = (idx + fracWithinStep) / total;
    const pct = Math.min(99, Math.round(overallFrac * 100));
    $("progressFill").style.width = `${pct}%`;

    $("progressStepLabel").textContent = data.current_step_name || data.progress_msg || "Working…";
    const subText = data.sub_progress && data.sub_progress.total
      ? ` (${data.sub_progress.current}/${data.sub_progress.total})`
      : "";
    $("progressStepCount").textContent = `Step ${idx + 1} of ${total}${subText}`;
  }

  function appendLog(msg, isErr) {
    const box = $("logBox");
    const line = document.createElement("div");
    line.className = "line" + (isErr ? " err" : "");
    line.textContent = msg;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  let seenLogCount = 0;
  async function pollJob() {
    if (!state.jobId) return;
    try {
      const res = await fetch(`/api/job/${state.jobId}`);
      const data = await res.json();

      const newLines = data.log.slice(seenLogCount);
      seenLogCount = data.log.length;
      for (const l of newLines) appendLog(l.msg);

      updateProgress(data);

      if (data.status === "running") {
        state.pollTimer = setTimeout(pollJob, 1200);
      } else if (data.status === "done") {
        $("runBtn").disabled = false;
        state.result = data.result;
        renderResults(data.result);
      } else if (data.status === "error") {
        appendLog(`Failed: ${data.error}`, true);
        $("runBtn").disabled = false;
      }
    } catch (err) {
      appendLog(String(err), true);
      state.pollTimer = setTimeout(pollJob, 2000);
    }
  }

  // ---------- results ----------
  function renderResults(result) {
    $("results").classList.add("active");

    const passBanner = $("verdictBanner");
    if (result.passes_both_after) {
      passBanner.className = "verdict-banner pass";
      $("verdictIcon").textContent = "✓";
      $("verdictTitle").textContent = "Passes both detector models";
      $("verdictSub").textContent = `Linear ${result.scores_after.linear_pct.toFixed(3)}% · CNN ${result.scores_after.cnn_pct.toFixed(1)}% · SNR ${result.overall_snr_db ? result.overall_snr_db.toFixed(1) + "dB" : "n/a"}`;
    } else {
      passBanner.className = "verdict-banner fail";
      $("verdictIcon").textContent = "!";
      $("verdictTitle").textContent = "Still flagged by at least one model";
      $("verdictSub").textContent = `Linear ${result.scores_after.linear_pct.toFixed(3)}% · CNN ${result.scores_after.cnn_pct.toFixed(1)}% — try selecting the AI-detector fix tools`;
    }

    const rows = [
      ["Linear model score", `${result.scores_before.linear_pct.toFixed(3)}%`, `${result.scores_after.linear_pct.toFixed(3)}%`, result.scores_after.passes_linear],
      ["CNN model score", `${result.scores_before.cnn_pct.toFixed(1)}%`, `${result.scores_after.cnn_pct.toFixed(1)}%`, result.scores_after.passes_cnn],
      ["Integrated LUFS", `${result.lufs_before.toFixed(1)}`, `${result.lufs_after.toFixed(1)}`, null],
      ["Duration", fmtDuration(result.duration_sec), fmtDuration(result.duration_sec), null],
      ["Signal-to-noise (change)", "—", result.overall_snr_db ? `${result.overall_snr_db.toFixed(1)} dB` : "unchanged", null],
    ];
    $("compareTable").innerHTML = `
      <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Status</th></tr></thead>
      <tbody>
      ${rows.map(([k, before, after, pass]) => `
        <tr>
          <td>${k}</td><td>${before}</td><td>${after}</td>
          <td>${pass === null ? "—" : pass ? '<span class="pill good">pass</span>' : '<span class="pill crit">flagged</span>'}</td>
        </tr>`).join("")}
      </tbody>`;

    $("stepsList").innerHTML = result.steps.map(s => `
      <div class="step-item">
        <div class="dot"></div>
        <div class="txt">${s.label}${stepDetailText(s)}</div>
        <div class="time">${s.elapsed_sec}s</div>
      </div>`).join("") || `<div class="step-item"><div class="txt">No processing steps were run.</div></div>`;

    drawSpectrum($("spectrumCanvas"), result.spectrum_before.freqs, result.spectrum_before.psd_db,
                 result.spectrum_after.freqs, result.spectrum_after.psd_db);

    setupABPlayer(result);
  }

  function stepDetailText(s) {
    if (!s.applied) return " — no change needed";
    if (s.tool === "trim_silence") return ` — removed ${s.lead_ms}ms / ${s.trail_ms}ms`;
    if (s.tool === "dc_offset") return ` — corrected ${s.dc_l_before.toFixed(5)} / ${s.dc_r_before.toFixed(5)}`;
    if (s.tool === "normalize_lufs") return ` — ${s.lufs_before.toFixed(1)} → ${s.lufs_after.toFixed(1)} LUFS`;
    if (s.tool === "true_peak_limit") return ` — reduced ${Math.abs(s.gain_reduction_db).toFixed(1)}dB`;
    if (s.tool === "linear_fix" || s.tool === "cnn_fix") return ` — SNR ${s.snr_db.toFixed(1)}dB`;
    if (s.tool === "fix_transients") return ` — ${s.count} anomal${s.count === 1 ? "y" : "ies"} found`;
    if (s.tool === "fix_phase") return s.correlation_after !== undefined
      ? ` — correlation ${s.correlation_before.toFixed(2)} → ${s.correlation_after.toFixed(2)}`
      : "";
    if (s.tool === "multiband_compress") return ` — up to ${Math.abs(Math.min(...s.bands.map(b => b.max_reduction_db))).toFixed(1)}dB gentle reduction`;
    return "";
  }

  // ---------- A/B player ----------
  function setupABPlayer(result) {
    const audioEl = $("audioEl");
    const origUrl = `/api/audio/output_orig/${result.out_id}`;
    const fixedUrl = `/api/audio/output/${result.out_id}`;
    state.urls = { orig: origUrl, fixed: fixedUrl };
    state.abMode = "orig";
    audioEl.src = origUrl;

    $("abOriginal").classList.add("active");
    $("abFixed").classList.remove("active");

    const outputName = result.output_name || "fixed.wav";
    const origName = outputName.replace(/(\.wav)?$/i, "_original.wav");
    const fixedDownloadUrl = `${fixedUrl}?name=${encodeURIComponent(outputName)}`;
    const origDownloadUrl = `${origUrl}?name=${encodeURIComponent(origName)}`;

    $("downloadOrig").onclick = () => { window.location.href = origDownloadUrl; };
    $("downloadFixed").onclick = () => { window.location.href = fixedDownloadUrl; };

    async function switchTo(mode) {
      const wasPlaying = !audioEl.paused;
      const t = audioEl.currentTime;
      state.abMode = mode;
      audioEl.src = mode === "orig" ? origUrl : fixedUrl;
      audioEl.currentTime = t;
      if (wasPlaying) audioEl.play();
      $("abOriginal").classList.toggle("active", mode === "orig");
      $("abFixed").classList.toggle("active", mode === "fixed");
    }
    $("abOriginal").onclick = () => switchTo("orig");
    $("abFixed").onclick = () => switchTo("fixed");

    $("playBtn").onclick = () => {
      if (audioEl.paused) { audioEl.play(); $("playBtn").textContent = "⏸"; }
      else { audioEl.pause(); $("playBtn").textContent = "▶"; }
    };
    audioEl.onended = () => { $("playBtn").textContent = "▶"; };

    audioEl.ontimeupdate = () => {
      $("timeCur").textContent = fmtDuration(audioEl.currentTime);
      if (audioEl.duration) {
        $("playhead").style.left = `${(audioEl.currentTime / audioEl.duration) * 100}%`;
      }
    };
    audioEl.onloadedmetadata = () => {
      $("timeTotal").textContent = fmtDuration(audioEl.duration);
    };

    const timeline = $("timeline");
    timeline.onclick = (e) => {
      const rect = timeline.getBoundingClientRect();
      const frac = (e.clientX - rect.left) / rect.width;
      if (audioEl.duration) audioEl.currentTime = frac * audioEl.duration;
    };

    drawWaveformPlaceholder();
  }

  function drawWaveformPlaceholder() {
    const canvas = $("waveCanvas");
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim();
    ctx.strokeStyle = accent;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    const bars = 140;
    for (let i = 0; i < bars; i++) {
      const x = (i / bars) * w;
      const seed = Math.sin(i * 12.9898) * 43758.5453;
      const amp = (Math.abs(seed - Math.floor(seed)) * 0.7 + 0.15) * (h / 2 - 2);
      ctx.beginPath();
      ctx.moveTo(x, h / 2 - amp);
      ctx.lineTo(x, h / 2 + amp);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
})();
