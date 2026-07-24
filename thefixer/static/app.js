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

  // ---------- docs overlay ----------
  const docsOverlay = $("docsOverlay");
  $("docsBtn").addEventListener("click", () => docsOverlay.classList.add("active"));
  $("docsCloseBtn").addEventListener("click", () => docsOverlay.classList.remove("active"));
  docsOverlay.addEventListener("click", (e) => {
    if (e.target === docsOverlay) docsOverlay.classList.remove("active");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") docsOverlay.classList.remove("active");
  });

  // ---------- info modal ----------
  const INFO_CONTENT = {
    linear_passes: {
      title: "What is the \"linear model\"?",
      body: `<p>One of two separate AI-detectors this tool checks your file against. It looks at the track's overall frequency spectrum for a specific pattern of small spectral spikes - a "fingerprint" some AI music generators leave behind as a byproduct of how they synthesize audio. A simple statistical model (logistic regression - hence "linear") scores how strongly that pattern is present, from 0% (no pattern found) to 100% (pattern strongly present).</p>
             <p><strong>Why it runs more than once:</strong> each attempt is checked against the real detector after it finishes, not just an internal estimate. If the real score is still above 1%, it retries with a stricter target, up to 4 times. It also has to survive being resampled up to the file's real sample rate before delivery, which can quietly undo some of the fix - so an attempt that looked perfect internally can still fail the final check and need another try.</p>`,
    },
    cnn_passes: {
      title: "What is the \"CNN model\"?",
      body: `<p>The second of two AI-detectors this tool checks against - a neural network (CNN = convolutional neural network) that looks at a more detailed time-frequency representation of the audio (a "CQT cepstrum," a way of laying out pitch and texture over time) and learned, from training examples, what AI-generated audio tends to look like in that representation. It's a deeper, more pattern-based check than the linear model's simpler frequency-fingerprint approach - and, in practice, meaningfully harder to correct without introducing audible change.</p>
             <p><strong>Why it shows so many step counts:</strong> it lays a new 10-second analysis window down every 2.5 seconds across the ENTIRE track, so the count scales with the song's length - a 3-minute track gets around 65-70 overlapping windows, a 5-minute track around 105-110. All of them share one correction, optimized together. Every 10-15 steps it re-checks itself against the real detector to see which windows still fail, and pushes harder on just those - that's the periodic "X/Y windows still above target" updates you see during a single pass.</p>`,
    },
    snr: {
      title: "Signal-to-noise ratio (SNR)",
      body: `<p>Measures how much the file changed, in decibels - specifically, how loud the original track is compared to how loud the <em>correction</em> added on top of it is.</p>
             <p>Higher = less change, i.e. the fix is quieter and less noticeable. It's not a measure of the file's own quality - it only exists as a before-vs-after comparison, which is why there's no separate "before" value.</p>
             <div class="good-bad">
               <div class="good"><strong>50dB+</strong><br>Very quiet correction, unlikely to be audible</div>
               <div class="bad"><strong>&lt;30dB</strong><br>A large, more noticeable change</div>
             </div>`,
    },
    lufs: {
      title: "LUFS (loudness)",
      body: `<p>Integrated loudness - roughly "how loud does this track feel overall," averaged across its whole length, following the same standard streaming platforms use to normalize playback volume.</p>
             <p>The tool targets <strong>-14 LUFS</strong>, the common streaming-platform reference (Spotify, YouTube, etc. all normalize toward something in this range) - loud enough to compete, without triggering automatic turn-down on playback.</p>
             <div class="good-bad">
               <div class="good"><strong>-16 to -12 LUFS</strong><br>Close to target, translates well across platforms</div>
               <div class="bad"><strong>Below -20 or above -8</strong><br>Notably quiet or aggressively loud/over-compressed</div>
             </div>`,
    },
    stereo_correlation: {
      title: "Stereo correlation",
      body: `<p>Measures how similar the left and right channels are, from -1 to +1. It's a check for phase problems - content that's out of phase between channels can partially or fully cancel out when played back in mono (phone speakers, some Bluetooth, club sound systems).</p>
             <div class="good-bad">
               <div class="good"><strong>+0.3 to +1.0</strong><br>Channels agree - safe in mono, no cancellation risk</div>
               <div class="bad"><strong>Below +0.1, especially negative</strong><br>Channels are fighting each other - real risk of thinning out or disappearing in mono</div>
             </div>`,
    },
    spectral_tilt: {
      title: "Spectral tilt (low/mid/high balance)",
      body: `<p>The average energy (in dB) in three frequency bands - low (20-250Hz, bass/sub), mid (250Hz-4kHz, where most instruments and vocals sit), and high (4kHz-20kHz, air/brightness/detail).</p>
             <p>This isn't about hitting exact numbers - it's about whether the balance across the three looks like a normal mix rather than something obviously lopsided.</p>
             <div class="good-bad">
               <div class="good">Good: mid is usually the strongest band, low and high taper off gradually - a smooth, natural-looking slope</div>
               <div class="bad">Bad: a sharp cliff in the high band (often means a hard cutoff, common in low-quality AI generation or lossy re-encoding), or bass so dominant it buries everything else</div>
             </div>`,
    },
    dc_offset: {
      title: "DC offset",
      body: `<p>A waveform should be centered on zero - equal amounts of positive and negative signal. DC offset means it's shifted up or down instead, which wastes headroom, can cause clicks when a file starts/stops, and adds no audible content of its own.</p>
             <div class="good-bad">
               <div class="good"><strong>Under 0.001</strong><br>Effectively centered, no meaningful offset</div>
               <div class="bad"><strong>Above 0.01</strong><br>Noticeably shifted - worth correcting</div>
             </div>`,
    },
    silence_trim: {
      title: "Leading/trailing silence",
      body: `<p>True silence (not just quiet - actual near-zero signal) at the very start or end of the file. This gets trimmed because it doesn't do anything for the listener, can trip up loudness measurement (a long silent intro drags down the average), and some platforms flag it as an upload error.</p>
             <p>Only genuine silence is touched - a quiet fade-in or fade-out is left alone.</p>`,
    },
    transients: {
      title: "Transients / pops detected",
      body: `<p>Sharp, sudden spikes in the waveform that don't fit the surrounding music - usually a click, pop, or edit artifact rather than an intentional musical hit. The tool only touches the exact moment flagged, gently limiting just that spike rather than compressing the whole track.</p>`,
    },
    tool_strip_metadata: {
      title: "Strip metadata & embedded images",
      body: `<p>Audio files can carry hidden text fields (title, artist, comments, "encoded by," generation timestamps) and even an embedded cover-art image, all invisible unless you go looking. Some AI platforms write their own name or a generation ID directly into these fields.</p>
             <p>This step reads and removes all of it. Every file this tool delivers is freshly re-encoded from raw audio anyway, so none of that original metadata could survive even if this step were skipped - it's shown here mainly so you can see exactly what was found.</p>`,
    },
    tool_trim_silence: {
      title: "Trim leading/trailing silence",
      body: `<p>Removes genuine silence (not just quiet passages) from the very start and end of the file. Long silent stretches can throw off loudness measurement and some platforms flag them as upload errors. A real fade-in or fade-out is left untouched - only true digital silence gets cut.</p>`,
    },
    tool_fix_transients: {
      title: "Surgical transient/pop fix",
      body: `<p>Scans for sharp, sudden clicks or pops - the kind that come from a bad edit, a glitch, or a digital artifact, not from actual instruments. Real percussive hits (kicks, snares) have a fast but natural attack and are deliberately left alone; this only touches genuine discontinuities.</p>
             <p>When found, only that exact moment is gently limited - the rest of the track is untouched.</p>`,
    },
    tool_spectral_revive: {
      title: "High-frequency fill-in (17kHz+)",
      body: `<p>Checks whether the track has an unnaturally hard cutoff somewhere above 17kHz - common when audio has been through lossy compression, or in some lower-quality AI generation, both of which can throw away high-frequency detail a human recording naturally has.</p>
             <p>If a real cutoff is found, this fills the missing top end back in - using only this specific track's own rolloff slope, its own detected harmonics, and its own dynamics as the reference. Nothing is borrowed from another file or a generic template.</p>`,
    },
    tool_high_pass: {
      title: "High-pass filter",
      body: `<p>Removes inaudible rumble below about 30Hz - content you can't hear but that still eats into your available headroom and can make a mix sound muddier than it needs to. Doesn't touch anything in the audible range.</p>`,
    },
    tool_fix_phase: {
      title: "Stereo phase / correlation correction",
      body: `<p>Checks whether the left and right channels are working together or fighting each other (see the Stereo correlation explanation for the actual measurement). If content is meaningfully out of phase, it can partially cancel out when the track is played back in mono - this step corrects that so the track stays safe on any playback system.</p>`,
    },
    tool_multiband_compress: {
      title: "Multiband tonal-balance compression",
      body: `<p>Gently compresses three frequency bands (low/mid/high) somewhat independently, rather than the whole signal at once - a light touch aimed at smoothing out any band that's poking out too far relative to the others, for a more balanced overall tone. Deliberately conservative, not a loudness-maximizing effect.</p>`,
    },
    tool_true_peak_limit: {
      title: "True-peak limiter",
      body: `<p>A final safety ceiling at -1dBTP (true-peak decibels, accounting for peaks that can appear between samples after digital-to-analog conversion, not just the samples themselves). This is the last line of defense against clipping/distortion on playback systems that are sensitive to true peaks, like streaming platforms and hardware DACs.</p>
             <p>It uses real dynamics limiting (pulling down only the moments that actually exceed the ceiling), not a flat volume reduction - so it doesn't quietly undo a loudness target that was already correctly set elsewhere in the chain.</p>`,
    },
  };

  function infoBtn(key) {
    return `<button class="info-btn" data-info="${key}" title="What's this?" type="button">i</button>`;
  }

  const infoOverlay = $("infoOverlay");
  function openInfo(key) {
    const c = INFO_CONTENT[key];
    if (!c) return;
    $("infoModalTitle").textContent = c.title;
    $("infoModalBody").innerHTML = c.body;
    infoOverlay.classList.add("active");
  }
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".info-btn");
    if (btn) { e.stopPropagation(); openInfo(btn.dataset.info); }
  });
  $("infoCloseBtn").addEventListener("click", () => infoOverlay.classList.remove("active"));
  infoOverlay.addEventListener("click", (e) => {
    if (e.target === infoOverlay) infoOverlay.classList.remove("active");
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") infoOverlay.classList.remove("active");
  });

  // ---------- tool catalog ----------
  const TOOLS = [
    { id: "strip_metadata", group: "chainGroupCleanup", name: "Strip metadata & embedded images", desc: "Reports and removes ID3/container tags (title, artist, comments, generation-platform provenance) and embedded cover art. The delivered file never carries these regardless, since every output is freshly encoded from raw audio - this step surfaces exactly what was found.", info: "tool_strip_metadata" },
    { id: "trim_silence", group: "chainGroupCleanup", name: "Trim silence", desc: "Removes leading/trailing true silence at the very start and end.", info: "tool_trim_silence" },
    { id: "dc_offset", group: "chainGroupCleanup", name: "DC offset correction", desc: "Centers the waveform on zero if it's biased up or down.", info: "dc_offset" },
    { id: "fix_transients", group: "chainGroupCleanup", name: "Surgical transient/pop fix", desc: "Auto-detects sharp pops/spikes and gently limits just that moment.", info: "tool_fix_transients" },
    { id: "spectral_revive", group: "chainGroupCleanup", name: "High-frequency fill-in (17kHz+)", desc: "Detects an artificial cutoff (common in lossy encoding or low-quality AI generation) and fills content above it using only this track's own rolloff slope, harmonics, and dynamics - no external reference.", info: "tool_spectral_revive" },
    { id: "high_pass", group: "chainGroupCleanup", name: "High-pass filter", desc: "Removes inaudible sub-30Hz rumble that eats into headroom.", info: "tool_high_pass" },
    { id: "linear_fix", group: "chainGroupAI", name: "Linear model fix", desc: "Gradient-optimized correction targeting the fakeprint logistic-regression detector.", info: "linear_passes" },
    { id: "cnn_fix", group: "chainGroupAI", name: "CNN model fix", desc: "Whole-track joint optimization targeting the CQT-cepstrum CNN detector. Slower.", info: "cnn_passes" },
    { id: "fix_phase", group: "chainGroupMaster", name: "Stereo phase correction", desc: "Corrects out-of-phase content that would cancel out in mono playback.", info: "tool_fix_phase" },
    { id: "normalize_lufs", group: "chainGroupMaster", name: "LUFS loudness normalization", desc: "Targets -14 LUFS, the standard streaming-platform loudness reference.", info: "lufs" },
    { id: "multiband_compress", group: "chainGroupMaster", name: "Multiband compression", desc: "Gentle 3-band dynamics smoothing for tonal balance.", info: "tool_multiband_compress" },
    { id: "true_peak_limit", group: "chainGroupMaster", name: "True-peak limiter", desc: "Brick-wall safety ceiling at -1dBTP, accounting for inter-sample peaks.", info: "tool_true_peak_limit" },
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
    outputFormat: "same",
    mp3Mode: "vbr0",
  };

  // ---------- output format switch ----------
  document.querySelectorAll("#formatSwitch button").forEach(btn => {
    btn.addEventListener("click", () => {
      state.outputFormat = btn.dataset.format;
      document.querySelectorAll("#formatSwitch button").forEach(b => b.classList.toggle("active", b === btn));
      $("mp3ModeRow").classList.toggle("hidden", btn.dataset.format !== "mp3");
    });
  });

  document.querySelectorAll("#mp3ModeSwitch button").forEach(btn => {
    btn.addEventListener("click", () => {
      state.mp3Mode = btn.dataset.mp3mode;
      document.querySelectorAll("#mp3ModeSwitch button").forEach(b => b.classList.toggle("active", b === btn));
    });
  });

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
    state = { ...state, fileId: null, analysis: null, selected: new Set(), jobId: null, result: null,
              lastResult: null, spectrumViewMode: "both", waveformViewMode: "both" };
    $("workspace").classList.remove("active");
    $("results").classList.remove("active");
    $("progressPanel").classList.remove("active");
    $("spectrumLegend").classList.remove("active");
    $("waveformLegend").classList.remove("active");
    $("analyzingState").classList.remove("active");
    document.querySelectorAll(".chart-view-row").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".chart-view-switch button").forEach(b => b.classList.toggle("active", b.dataset.view === "both"));
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

      const fmt = data.source_format || {};
      const fmtParts = [];
      if (fmt.codec) fmtParts.push(fmt.codec);
      if (fmt.sample_rate_hz) fmtParts.push(`${(fmt.sample_rate_hz / 1000).toFixed(1)}kHz`);
      if (fmt.bit_depth) fmtParts.push(`${fmt.bit_depth}-bit`);
      else if (fmt.bit_rate_kbps) fmtParts.push(`${fmt.bit_rate_kbps}kbps`);
      if (fmt.channels) fmtParts.push(fmt.channels === 2 ? "stereo" : fmt.channels === 1 ? "mono" : `${fmt.channels}ch`);
      $("fileFormatMeta").textContent = fmtParts.join(" · ");

      const origPlayer = $("originalPlayer");
      origPlayer.src = `/api/audio/upload/${data.file_id}`;
      origPlayer.classList.add("active");

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
    $("analyzingState").classList.add("active");
    const startedAt = performance.now();
    const tick = setInterval(() => {
      const secs = Math.round((performance.now() - startedAt) / 1000);
      $("analyzingLabel").textContent = secs < 3
        ? "Analyzing audio (running AI detectors, spectral checks)…"
        : `Analyzing audio (running AI detectors, spectral checks)… ${secs}s`;
    }, 1000);
    try {
      const res = await fetch(`/api/analyze/${state.fileId}`);
      const data = await res.json();
      state.analysis = data;
      renderAnalysis(data);
      state.selected = new Set(data.recommended_tools);
      renderToolChain();
    } finally {
      clearInterval(tick);
      $("analyzingState").classList.remove("active");
    }
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

    // status: "good" | "warn" | "bad" | null (null = no pill, purely informational)
    const dcMax = Math.max(Math.abs(data.dc_offset.l), Math.abs(data.dc_offset.r));
    const stats = [
      {
        k: "LUFS (integrated)", v: `${data.lufs.toFixed(1)} LU`, info: "lufs",
        status: (data.lufs >= -16 && data.lufs <= -12) ? "good" : (data.lufs < -20 || data.lufs > -8) ? "warn" : null,
      },
      {
        k: "Stereo correlation", v: data.stereo_correlation.toFixed(2), info: "stereo_correlation",
        status: data.stereo_correlation >= 0.3 ? "good" : data.stereo_correlation < 0.1 ? "bad" : "warn",
      },
      {
        k: "DC offset (L/R)", v: `${data.dc_offset.l.toFixed(5)} / ${data.dc_offset.r.toFixed(5)}`, info: "dc_offset",
        status: dcMax < 0.001 ? "good" : dcMax > 0.01 ? "warn" : null,
      },
      {
        k: "Leading/trailing silence", v: `${data.silence.lead_ms || 0}ms / ${data.silence.trail_ms || 0}ms`, info: "silence_trim",
        status: ((data.silence.lead_ms || 0) > 0 || (data.silence.trail_ms || 0) > 0) ? "warn" : "good",
      },
      {
        k: "Transients detected", v: data.transients.length, info: "transients",
        status: data.transients.length > 0 ? "warn" : "good",
      },
      {
        k: "Spectral tilt (low/mid/high)", v: `${data.spectral_tilt["low (20-250Hz)"].toFixed(0)} / ${data.spectral_tilt["mid (250-4000Hz)"].toFixed(0)} / ${data.spectral_tilt["high (4000-20000Hz)"].toFixed(0)} dB`, info: "spectral_tilt",
        status: null,
      },
    ];
    const statusLabel = { good: "Safe", warn: "Check", bad: "Risk" };
    $("statList").innerHTML = stats.map(s => `
      <div class="stat">
        <div class="k">${s.k}${infoBtn(s.info)}</div>
        <div class="row">
          <div class="v">${s.v}</div>
          ${s.status ? `<span class="pill small ${s.status === "bad" ? "crit" : s.status}">${statusLabel[s.status]}</span>` : ""}
        </div>
      </div>`).join("");

    drawSpectrum($("spectrumCanvas"), data.spectrum.freqs, data.spectrum.psd_db);
    drawWaveformOverview($("waveformOverviewCanvas"), data.waveform);
    $("waveformDuration").textContent = fmtDuration(data.waveform.duration_sec);
    $("waveformLegend").classList.remove("active");
  }

  function drawSpectrum(canvas, freqs, psdDb, overlayFreqs, overlayPsd, mode = "both") {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim();
    const before = styles.getPropertyValue("--before").trim();
    const rackLine = styles.getPropertyValue("--rack-line").trim();
    const textFaint = styles.getPropertyValue("--text-faint").trim();

    const minDb = -100, maxDb = 20;
    const logMin = Math.log10(20), logMax = Math.log10(20000);

    // grid + dB axis labels - power spectral density in dB (higher = more
    // energy at that frequency); without these labels the vertical axis
    // has no unit at all, only the horizontal (Hz) does
    ctx.strokeStyle = rackLine;
    ctx.lineWidth = 1;
    ctx.font = "9px ui-monospace, monospace";
    ctx.fillStyle = textFaint;
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 4; i++) {
      const y = (h / 4) * i;
      if (i > 0 && i < 4) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      const db = maxDb - (i / 4) * (maxDb - minDb);
      ctx.fillText(`${Math.round(db)}dB`, 4, i === 0 ? y + 7 : (i === 4 ? y - 5 : y));
    }

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

    if (overlayFreqs && mode === "both") {
      const beforePts = pathFor(freqs, psdDb);
      drawLine(beforePts, before, 1.75);
      const afterPts = pathFor(overlayFreqs, overlayPsd);
      drawFill(afterPts, accent, 0.12);
      drawLine(afterPts, accent, 2);
    } else if (overlayFreqs && mode === "before") {
      const pts = pathFor(freqs, psdDb);
      drawFill(pts, before, 0.15);
      drawLine(pts, before, 1.75);
    } else if (overlayFreqs && mode === "after") {
      const pts = pathFor(overlayFreqs, overlayPsd);
      drawFill(pts, accent, 0.15);
      drawLine(pts, accent, 1.75);
    } else {
      const pts = pathFor(freqs, psdDb);
      drawFill(pts, accent, 0.15);
      drawLine(pts, accent, 1.75);
    }
  }

  function drawWaveformOverview(canvas, wave, overlayWave, mode = "both", markers = null) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim();
    const before = styles.getPropertyValue("--before").trim();
    const rackLine = styles.getPropertyValue("--rack-line").trim();
    const textFaint = styles.getPropertyValue("--text-faint").trim();
    const midY = h / 2;

    // center line + amplitude axis labels (full scale is -1..1, the
    // normalized sample range) so the vertical axis has a clear unit,
    // same as the spectrum chart's dB labels
    ctx.strokeStyle = rackLine;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(w, midY); ctx.stroke();
    ctx.font = "9px ui-monospace, monospace";
    ctx.fillStyle = textFaint;
    ctx.textBaseline = "middle";
    ctx.fillText("+1.0", 4, 7);
    ctx.fillText("0.0", 4, midY);
    ctx.fillText("-1.0", 4, h - 7);

    function drawEnvelope(waveData, color, alpha) {
      if (!waveData || !waveData.max || !waveData.max.length) return;
      const n = waveData.max.length;
      const step = w / n;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = i * step;
        const y = midY - waveData.max[i] * midY;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      for (let i = n - 1; i >= 0; i--) {
        const x = i * step;
        const y = midY - waveData.min[i] * midY;
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = Math.min(1, alpha * 4);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (overlayWave && mode === "both") {
      drawEnvelope(wave, before, 0.35);
      drawEnvelope(overlayWave, accent, 0.35);
    } else if (overlayWave && mode === "before") {
      drawEnvelope(wave, before, 0.45);
    } else if (overlayWave && mode === "after") {
      drawEnvelope(overlayWave, accent, 0.45);
    } else {
      drawEnvelope(wave, accent, 0.4);
    }

    // transient/pop flags: small triangle markers at each detected glitch's
    // exact timestamp, so it's visible ON the waveform where a problem was
    // found, not just as a bare count in the stats panel
    if (markers && markers.length) {
      const crit = styles.getPropertyValue("--crit").trim();
      const durationSec = (wave && wave.duration_sec) || (overlayWave && overlayWave.duration_sec) || 0;
      if (durationSec > 0) {
        ctx.fillStyle = crit;
        for (const m of markers) {
          const x = (m.time_sec / durationSec) * w;
          ctx.beginPath();
          ctx.moveTo(x - 4, 0);
          ctx.lineTo(x + 4, 0);
          ctx.lineTo(x, 7);
          ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = crit;
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.5;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, h);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
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
            <div class="name">${t.name}${t.info ? infoBtn(t.info) : ""}${recommended ? `<span class="rec-badge">Recommended</span>` : ""}</div>
            <div class="desc">${t.desc}</div>
          </div>
        </div>`;
      }).join("");
    }

    document.querySelectorAll(".tool-row").forEach(row => {
      row.addEventListener("click", (e) => {
        // clicking the info (i) button should only open its popup, not
        // also toggle the tool's checkbox - this listener is attached
        // directly on the row (fires before the global document-level
        // info-btn handler even sees the bubbled event), so stopPropagation
        // inside that later handler is too late to prevent this one from
        // already running. Bail out here directly instead.
        if (e.target.closest(".info-btn")) return;
        const id = row.dataset.tool;
        if (state.selected.has(id)) state.selected.delete(id);
        else state.selected.add(id);
        renderToolChain();
      });
    });

    const countIds = { chainGroupCleanup: "countCleanup", chainGroupAI: "countAI", chainGroupMaster: "countMaster" };
    for (const [groupId, tools] of Object.entries(groups)) {
      const checkedCount = tools.filter(t => state.selected.has(t.id)).length;
      $(countIds[groupId]).textContent = `${checkedCount}/${tools.length}`;
    }

    $("runCount").textContent = `${state.selected.size} selected`;
    $("checkAllBtn").textContent = TOOLS.every(t => state.selected.has(t.id)) ? "Uncheck all" : "Check all";
  }

  $("selectRecBtn").addEventListener("click", () => {
    if (!state.analysis) return;
    state.selected = new Set(state.analysis.recommended_tools);
    renderToolChain();
  });

  $("checkAllBtn").addEventListener("click", () => {
    const allChecked = TOOLS.every(t => state.selected.has(t.id));
    state.selected = allChecked ? new Set() : new Set(TOOLS.map(t => t.id));
    renderToolChain();
  });

  document.querySelectorAll(".chain-group-header").forEach(btn => {
    btn.addEventListener("click", () => {
      const expanded = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", expanded ? "false" : "true");
    });
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
      body: JSON.stringify({
        tools: Array.from(state.selected), options: {},
        output_name: outputName || undefined,
        output_format: state.outputFormat,
        mp3_mode: state.mp3Mode,
      }),
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
    let subText = "";
    if (data.sub_progress && data.sub_progress.total) {
      const sp = data.sub_progress;
      subText = ` (optimization step ${sp.current} of ${sp.total}`;
      if (sp.attempt && sp.max_attempts) subText += `, attempt ${sp.attempt}/${sp.max_attempts}`;
      if (sp.score_pct !== undefined) subText += `, live estimate ${sp.score_pct}% AI`;
      if (sp.windows_total !== undefined) {
        subText += `, real-model check: ${sp.windows_failing}/${sp.windows_total} windows still above target (${sp.real_max_score_pct}% AI)`;
      }
      subText += ")";
    }
    $("progressStepCount").textContent = `Step ${idx + 1} of ${total}${subText}`;
  }

  // translates the technical pipeline log into a plain-language line for
  // the primary display, while keeping the raw text available (small,
  // dimmed, underneath) for anyone who wants the exact detail - rather
  // than rewriting the raw strings themselves, which stay precise for
  // debugging.
  const LOG_TRANSLATIONS = [
    [/^loading .+$/, () => "Loading your file…"],
    [/^running: (.+)$/, (m) => `Starting: ${m[1]}`],
    [/^\s*done \(([\d.]+)s\)$/, (m) => `Done (${m[1]}s)`],
    [/^linear: attempt (\d+) of (\d+) - optimizing.*$/, (m) => `AI-detector fix (linear model): trying attempt ${m[1]} of ${m[2]}…`],
    [/^linear: attempt \d+ result checked against the REAL detector.*: ([\d.]+)%$/, (m) => {
      const pct = Number(m[1]);
      return { text: `Checked against the real detector: ${m[1]}% AI-likely`, badge: pct < 1 ? "pass" : "retry" };
    }],
    [/^linear: real score [\d.]+% is above the <1% target - retrying.*$/, () => ({ text: "Not quite under 1% yet - trying again with a stricter setting", badge: "retry" })],
    [/^linear: real score [\d.]+% is above the <1% target after all \d+ attempts.*$/, () => ({ text: "Didn't fully clear 1% after all attempts - shipping the best result found", badge: "fail" })],
    [/^cnn: optimizing ([\d.]+)s of audio.*$/, (m) => `AI-detector fix (CNN model): working through ${Math.round(m[1])}s of audio - this one takes a while…`],
    [/^cnn: verifying final transferred stereo output against the real model$/, () => "Double-checking the CNN fix on the actual file you'll receive"],
    [/^re-verifying linear model after full chain.*$/, () => "Double-checking the linear fix still holds after mastering"],
    [/^\s*post-chain linear score: ([\d.]+)%$/, (m) => {
      const pct = Number(m[1]);
      return { text: `Linear model result: ${m[1]}% AI-likely`, badge: pct < 1 ? "pass" : "retry" };
    }],
    [/^\s*above target - re-running linear_fix.*$/, () => ({ text: "Still above target - running the linear fix once more", badge: "retry" })],
    [/^\s*post-chain cnn score: ([\d.]+)%$/, (m) => {
      const pct = Number(m[1]);
      return { text: `CNN model result: ${m[1]}% AI-likely`, badge: pct < 8 ? "pass" : "retry" };
    }],
    [/^\s*cnn lost its safety margin.*$/, () => ({ text: "The CNN fix slipped after a later step - running it again", badge: "retry" })],
    [/^re-running true-peak limiter.*$/, () => "Re-checking the loudness ceiling after that last change"],
    [/^post-chain LUFS check: ([\-\d.]+) vs target ([\-\d.]+).*$/, (m) => `Loudness drifted to ${m[1]} (target ${m[2]}) - correcting`],
    [/^\s*corrected to ([\-\d.]+) LUFS.*$/, (m) => `Loudness corrected to ${m[1]}`],
    [/^saving output file.*$/, () => "Saving your finished file…"],
    [/^re-scoring with AI detectors$/, () => "Running the final check with both AI detectors"],
    [/^WARNING: final file still flagged by at least one model \((.+)\)$/, (m) => ({ text: `Heads up: still flagged by at least one detector (${m[1]})`, badge: "fail" })],
    [/^\s*WARNING: linear regressed.*$/, () => ({ text: "Heads up: the linear score slipped a bit after a later step", badge: "fail" })],
    [/^\s*WARNING: delivered file is ([\-\d.]+) LUFS.*$/, (m) => `Heads up: couldn't fully reach the loudness target (landed at ${m[1]} LUFS) without exceeding the peak safety ceiling`],
    [/^complete$/, () => "All done"],
    [/^ERROR: (.+)$/, (m) => `Something went wrong: ${m[1]}`],
  ];

  function friendlyLog(msg) {
    for (const [pattern, fn] of LOG_TRANSLATIONS) {
      const m = msg.match(pattern);
      if (m) return fn(m);
    }
    return null;
  }

  const LOG_BADGE_LABEL = { pass: "PASS", retry: "RETRY", fail: "FLAGGED" };

  function appendLog(msg, isErr) {
    const box = $("logBox");
    const line = document.createElement("div");
    line.className = "line" + (isErr ? " err" : "");
    const result = friendlyLog(msg);
    const friendlyText = result && typeof result === "object" ? result.text : result;
    const badge = result && typeof result === "object" ? result.badge : null;
    if (friendlyText && friendlyText !== msg) {
      const main = document.createElement("div");
      main.className = "line-main";
      const textSpan = document.createElement("span");
      textSpan.textContent = friendlyText;
      main.appendChild(textSpan);
      if (badge) {
        const badgeSpan = document.createElement("span");
        badgeSpan.className = `log-badge ${badge}`;
        badgeSpan.textContent = LOG_BADGE_LABEL[badge] || badge;
        main.appendChild(badgeSpan);
      }
      const raw = document.createElement("div");
      raw.className = "line-raw";
      raw.textContent = msg;
      line.appendChild(main);
      line.appendChild(raw);
    } else {
      line.textContent = msg;
    }
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
      ["Linear model score", `${result.scores_before.linear_pct.toFixed(3)}%`, `${result.scores_after.linear_pct.toFixed(3)}%`, result.scores_after.passes_linear, "linear_passes"],
      ["CNN model score", `${result.scores_before.cnn_pct.toFixed(1)}%`, `${result.scores_after.cnn_pct.toFixed(1)}%`, result.scores_after.passes_cnn, "cnn_passes"],
      ["Integrated LUFS", `${result.lufs_before.toFixed(1)}`, `${result.lufs_after.toFixed(1)}`, null, "lufs"],
      ["Duration", fmtDuration(result.duration_sec), fmtDuration(result.duration_sec), null, null],
      ["Signal-to-noise (original vs. fixed)", "n/a (reference)", result.overall_snr_db ? `${result.overall_snr_db.toFixed(1)} dB` : "unchanged", null, "snr"],
    ];
    $("compareTable").innerHTML = `
      <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Status</th></tr></thead>
      <tbody>
      ${rows.map(([k, before, after, pass, info]) => `
        <tr>
          <td>${k}${info ? infoBtn(info) : ""}</td><td>${before}</td><td>${after}</td>
          <td>${pass === null ? "—" : pass ? '<span class="pill good">pass</span>' : '<span class="pill crit">flagged</span>'}</td>
        </tr>`).join("")}
      </tbody>`;

    $("stepsList").innerHTML = result.steps.map(s => {
      const warning = stepWarningText(s);
      return `
      <div class="step-item">
        <div class="dot${warning ? " warn" : ""}"></div>
        <div class="txt">
          ${s.label}${stepDetailText(s)}
          ${warning ? `<div class="step-warning">⚠ ${warning}</div>` : ""}
        </div>
        <div class="time">${s.elapsed_sec}s</div>
      </div>`;
    }).join("") || `<div class="step-item"><div class="txt">No processing steps were run.</div></div>`;

    state.lastResult = result;
    state.spectrumViewMode = state.spectrumViewMode || "both";
    state.waveformViewMode = state.waveformViewMode || "both";

    renderSpectrumView();
    $("spectrumLegend").classList.add("active");
    $("spectrumLegend").innerHTML = `
      <div class="item"><span class="swatch" style="background: var(--before)"></span>Before</div>
      <div class="item"><span class="swatch" style="background: var(--accent)"></span>After</div>
    `;
    document.querySelector("#spectrumCanvas").closest(".spectrum-wrap").querySelector(".chart-view-row").classList.add("active");

    if (result.waveform_before && result.waveform_after) {
      renderWaveformView();
      $("waveformDuration").textContent = fmtDuration(result.waveform_after.duration_sec);
      $("waveformLegend").classList.add("active");
      $("waveformLegend").innerHTML = `
        <div class="item"><span class="swatch" style="background: var(--before)"></span>Before</div>
        <div class="item"><span class="swatch" style="background: var(--accent)"></span>After</div>
      `;
      document.querySelector("#waveformOverviewCanvas").closest(".spectrum-wrap").querySelector(".chart-view-row").classList.add("active");
    }

    setupABPlayer(result);
  }

  function renderSpectrumView() {
    const result = state.lastResult;
    if (!result) return;
    const mode = state.spectrumViewMode;
    drawSpectrum($("spectrumCanvas"), result.spectrum_before.freqs, result.spectrum_before.psd_db,
                 result.spectrum_after.freqs, result.spectrum_after.psd_db, mode);
  }

  function renderWaveformView() {
    const result = state.lastResult;
    if (!result || !result.waveform_before || !result.waveform_after) return;
    const mode = state.waveformViewMode;
    drawWaveformOverview($("waveformOverviewCanvas"), result.waveform_before, result.waveform_after, mode);
  }

  document.querySelectorAll("#spectrumViewSwitch button").forEach(btn => {
    btn.addEventListener("click", () => {
      state.spectrumViewMode = btn.dataset.view;
      document.querySelectorAll("#spectrumViewSwitch button").forEach(b => b.classList.toggle("active", b === btn));
      renderSpectrumView();
    });
  });
  document.querySelectorAll("#waveformViewSwitch button").forEach(btn => {
    btn.addEventListener("click", () => {
      state.waveformViewMode = btn.dataset.view;
      document.querySelectorAll("#waveformViewSwitch button").forEach(b => b.classList.toggle("active", b === btn));
      renderWaveformView();
    });
  });

  function stepDetailText(s) {
    if (s.tool === "strip_metadata") {
      if (!s.applied) return " — no metadata or embedded images found on the source file";
      const tagNames = Object.keys(s.tags_found || {});
      const parts = [];
      if (tagNames.length) parts.push(`removed tags: ${tagNames.join(", ")}`);
      if (s.has_embedded_images) parts.push("removed embedded cover art");
      return ` — ${parts.join("; ")}`;
    }
    if (!s.applied) return " — no change needed";
    let text = "";
    if (s.tool === "trim_silence") text = ` — removed ${s.lead_ms}ms / ${s.trail_ms}ms`;
    else if (s.tool === "dc_offset") text = ` — corrected ${s.dc_l_before.toFixed(5)} / ${s.dc_r_before.toFixed(5)}`;
    else if (s.tool === "normalize_lufs") text = ` — ${s.lufs_before.toFixed(1)} → ${s.lufs_after.toFixed(1)} LUFS`;
    else if (s.tool === "true_peak_limit") text = ` — reduced ${Math.abs(s.gain_reduction_db).toFixed(1)}dB`;
    else if (s.tool === "linear_fix" || s.tool === "linear_fix_reverify")
      text = ` — SNR ${s.snr_db.toFixed(1)}dB` + (s.final_real_score !== undefined ? `, ${(s.final_real_score * 100).toFixed(3)}% AI` : "");
    else if (s.tool === "cnn_fix") {
      text = ` — SNR ${s.snr_db.toFixed(1)}dB`;
      if (s.worst_score_after_transfer !== undefined && s.worst_score_after_transfer !== null) {
        text += `, worst of ${s.n_windows || "all"} optimization windows: ${(s.worst_score_after_transfer * 100).toFixed(1)}% AI`;
      }
    }
    else if (s.tool === "fix_transients") text = ` — ${s.count} anomal${s.count === 1 ? "y" : "ies"} found`;
    else if (s.tool === "spectral_revive") text = ` — filled above ${(s.cutoff_hz/1000).toFixed(0)}kHz (self-fitted rolloff: ${s.fitted_rolloff_db_per_octave.toFixed(1)}dB/octave)`;
    else if (s.tool === "fix_phase") text = s.correlation_after !== undefined
      ? ` — correlation ${s.correlation_before.toFixed(2)} → ${s.correlation_after.toFixed(2)}`
      : "";
    else if (s.tool === "multiband_compress") text = ` — up to ${Math.abs(Math.min(...s.bands.map(b => b.max_reduction_db))).toFixed(1)}dB gentle reduction`;

    if (s.triggered_by) text += ` (auto re-run: ${s.triggered_by})`;
    return text;
  }

  function stepWarningText(s) {
    // Surface the honesty-signal fields that mean "this fix didn't fully
    // reach its target" - these are set specifically so a partial fix is
    // never silently reported as a full success.
    if (s.warning) return s.warning;
    if (s.tool === "cnn_fix" && s.verified_after_transfer === false) {
      return `Not all analysis windows converged - worst window still scored ${(s.worst_score_after_transfer * 100).toFixed(1)}% AI after transfer.`;
    }
    return null;
  }

  // ---------- A/B player ----------
  // Both A and B are loaded into their own <audio> element simultaneously and
  // kept playing in lockstep at all times; switching just mutes one and
  // unmutes the other (instant, sample-accurate, no re-fetch/re-decode gap)
  // instead of swapping a single element's `src` mid-playback, which would
  // force the browser to re-buffer and produce an audible glitch at the
  // switch point.
  function setupABPlayer(result) {
    const audioOrig = $("audioOrig");
    const audioFixed = $("audioFixed");
    const origUrl = `/api/audio/output_orig/${result.out_id}`;
    const fixedUrl = `/api/audio/output/${result.out_id}`;
    state.urls = { orig: origUrl, fixed: fixedUrl };
    state.abMode = "orig";

    audioOrig.src = origUrl;
    audioFixed.src = fixedUrl;
    audioOrig.muted = false;
    audioFixed.muted = true;

    $("abOriginal").classList.add("active");
    $("abFixed").classList.remove("active");

    const outputName = result.output_name || "fixed.wav";
    const origName = outputName.replace(/(\.wav)?$/i, "_original.wav");
    const fixedDownloadUrl = `${fixedUrl}?name=${encodeURIComponent(outputName)}`;
    const origDownloadUrl = `${origUrl}?name=${encodeURIComponent(origName)}`;

    $("downloadOrig").onclick = () => { window.location.href = origDownloadUrl; };
    $("downloadFixed").onclick = () => { window.location.href = fixedDownloadUrl; };

    function activeEl() { return state.abMode === "orig" ? audioOrig : audioFixed; }
    function inactiveEl() { return state.abMode === "orig" ? audioFixed : audioOrig; }

    function switchTo(mode) {
      if (mode === state.abMode) return;
      state.abMode = mode;
      audioOrig.muted = mode !== "orig";
      audioFixed.muted = mode !== "fixed";
      // guard against any drift between the two elements' clocks
      const master = activeEl();
      const other = inactiveEl();
      if (Math.abs(other.currentTime - master.currentTime) > 0.05) {
        other.currentTime = master.currentTime;
      }
      $("abOriginal").classList.toggle("active", mode === "orig");
      $("abFixed").classList.toggle("active", mode === "fixed");
    }
    $("abOriginal").onclick = () => switchTo("orig");
    $("abFixed").onclick = () => switchTo("fixed");

    $("playBtn").onclick = () => {
      if (activeEl().paused) {
        audioOrig.currentTime = audioFixed.currentTime = activeEl().currentTime;
        audioOrig.play();
        audioFixed.play();
        $("playBtn").textContent = "⏸";
      } else {
        audioOrig.pause();
        audioFixed.pause();
        $("playBtn").textContent = "▶";
      }
    };
    audioOrig.onended = audioFixed.onended = () => { $("playBtn").textContent = "▶"; };

    $("restartBtn").onclick = () => {
      audioOrig.currentTime = 0;
      audioFixed.currentTime = 0;
      $("timeCur").textContent = fmtDuration(0);
      $("playhead").style.left = "0%";
    };

    let lastVolume = 1;
    function applyVolume(v) {
      audioOrig.volume = v;
      audioFixed.volume = v;
      $("muteBtn").textContent = v === 0 ? "🔇" : v < 0.5 ? "🔉" : "🔊";
    }
    $("volumeSlider").oninput = (e) => {
      const v = Number(e.target.value) / 100;
      applyVolume(v);
      if (v > 0) lastVolume = v;
    };
    $("muteBtn").onclick = () => {
      const isMuted = (audioOrig.volume === 0);
      if (isMuted) {
        applyVolume(lastVolume);
        $("volumeSlider").value = Math.round(lastVolume * 100);
      } else {
        lastVolume = audioOrig.volume || lastVolume;
        applyVolume(0);
        $("volumeSlider").value = 0;
      }
    };
    applyVolume(1);
    $("volumeSlider").value = 100;

    const onTimeUpdate = () => {
      const el = activeEl();
      $("timeCur").textContent = fmtDuration(el.currentTime);
      if (el.duration) {
        $("playhead").style.left = `${(el.currentTime / el.duration) * 100}%`;
      }
    };
    audioOrig.ontimeupdate = onTimeUpdate;
    audioFixed.ontimeupdate = onTimeUpdate;

    const onLoadedMetadata = () => {
      const el = activeEl();
      if (el.duration) $("timeTotal").textContent = fmtDuration(el.duration);
    };
    audioOrig.onloadedmetadata = onLoadedMetadata;
    audioFixed.onloadedmetadata = onLoadedMetadata;

    const timeline = $("timeline");
    timeline.onclick = (e) => {
      const rect = timeline.getBoundingClientRect();
      const frac = (e.clientX - rect.left) / rect.width;
      const el = activeEl();
      if (el.duration) {
        const t = frac * el.duration;
        audioOrig.currentTime = t;
        audioFixed.currentTime = t;
      }
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
