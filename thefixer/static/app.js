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
             <p><strong>Why it shows so many step counts:</strong> the recommended shift-robust mode repeatedly trains and checks the detector's five real 10-second evaluation positions across small timing shifts. Every 10 steps it re-checks against the real detector and reports how many positions still miss the safety target. Thorough mode instead tiles dense overlapping windows across the whole track and is substantially slower.</p>`,
    },
    temporal_normalize: {
      title: "What is \"temporal pattern denormalization\"?",
      body: `<p>The linear and CNN models above both look at the audio's spectral content - what frequencies are present. This is a different idea: some AI-generated audio can carry unnaturally precise, machine-regular timing underneath the music itself - a kind of rigid internal grid that's a byproduct of how the audio was generated, distinct from anything a spectral classifier looks at.</p>
             <p>This step smooths that timing very slightly and unevenly across the track - never more than a few milliseconds of drift at any single moment, varying slowly and smoothly rather than as one flat speed change (a flat change would be trivial to detect and undo; this instead nudges the internal timing map itself).</p>
             <p><strong>Why 4ms:</strong> fingerprint matching anchors on strong, sparse, low-frequency spectral peaks — measured on a real track, 94% sit below 500Hz and none above 4kHz. Landmark timing is quantized by the analysis hop (~11.6ms), so once the drift moves a landmark past one hop it cannot move further. Displacement saturates at 4ms; 8ms and 15ms produce identical landmark movement.</p>
             <p><strong>What higher values cost:</strong> resampling through a drifting time axis smears fast high-frequency content, which is where sibilants live — the "s" and "t" sounds in a vocal. At 4ms a measured sibilant lost 1.8% of its energy; at 15ms it lost 13.3%, concentrated in the 1-4kHz band while bass was untouched. Since that band contributes nothing to fingerprint matching, the extra drift is damage without benefit.</p>
             <p>This tool has no access to any commercial fingerprinting service, so its effect is measured against a local landmark-matching proxy built on the same constellation principle. Each run uses a fresh random warp. Off by default.</p>`,
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
             <p>When one is found, that exact moment is repaired by interpolating across it from the clean samples either side - the discontinuity is removed rather than just turned down, since simply reducing its level leaves the jump itself intact. The rest of the track is untouched.</p>
             <p><strong>What it will not touch:</strong> a click is near-instantaneous - one or two samples cross the detection threshold. A vocal consonant ("s", "t", "k") is a sustained broadband burst that crosses it hundreds of times, so bursts are rejected. Because the repair works by deleting, a false positive here would erase a consonant rather than merely duck it.</p>
             <p><strong>The post-chain re-check:</strong> after everything else has run, the track is scanned once more, because later stages can introduce their own artifacts. That second pass only corrects anomalies that were NOT in your source file. Compression and limiting smooth a consonant enough that it can start to look like a click, and deleting a sharp edge that was already in your recording is not this tool's job.</p>`,
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
      body: `<p>Gently compresses three frequency bands (low/mid/high) somewhat independently, rather than the whole signal at once - a light touch aimed at smoothing out any band that's poking out too far relative to the others, for a more balanced overall tone. Deliberately conservative, not a loudness-maximizing effect.</p>
             <p><strong>It repeats itself as needed.</strong> The gentle ratio only closes about a quarter of the gap each pass, so a strongly peaky track would need several passes to settle. Rather than asking you to run the file through again, it keeps passing until the measured imbalance stops improving (up to a bounded limit), then reports how many passes it took. A track that is already balanced is left completely untouched.</p>`,
    },
    tool_true_peak_limit: {
      title: "True-peak limiter",
      body: `<p>A final safety ceiling at -1dBTP (true-peak decibels, accounting for peaks that can appear between samples after digital-to-analog conversion, not just the samples themselves). This is the last line of defense against clipping/distortion on playback systems that are sensitive to true peaks, like streaming platforms and hardware DACs.</p>
             <p>It uses real dynamics limiting (pulling down only the moments that actually exceed the ceiling), not a flat volume reduction - so it doesn't quietly undo a loudness target that was already correctly set elsewhere in the chain.</p>`,
    },
    cnn_mode: {
      title: "CNN fix mode: Simple vs. Shift-robust vs. Thorough",
      body: `<p><strong>Simple</strong> optimizes ONLY the exact 5 fixed positions the real CNN detector itself checks (an even spread across the track, skipping the first/last 5 seconds) - what an off-the-shelf, uncustomized deployment of this detector would actually test against. Fastest, but the resulting fix can be fragile: it's tuned to that exact spot and can fail if the delivered file's exact alignment drifts by even a fraction of a second.</p>
             <p><strong>Shift-robust</strong> optimizes the same 5 positions across small (±0.5s) shifts. It is retained for comparison with the original optimizer.</p>
             <p><strong>Thorough</strong> (recommended, default) protects every 0.5-second whole-track window plus the detector's exact fractional test positions and timing neighborhoods. It evaluates exact windows in parallel, stops repeating gradient work on safe regions, and verifies the native delivered signal while the optimizer is still warm. A failed native check continues from the existing correction instead of restarting the track.</p>`,
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
    { id: "fix_transients", group: "chainGroupCleanup", name: "Surgical transient/pop fix", desc: "Finds genuine clicks and pops and bridges across just that moment. Skips sustained bursts like vocal consonants, and its post-chain re-check only corrects anomalies this chain introduced — never sharp edges that were already in your source.", info: "tool_fix_transients" },
    { id: "spectral_revive", group: "chainGroupCleanup", name: "High-frequency fill-in (17kHz+)", desc: "Detects an artificial cutoff (common in lossy encoding or low-quality AI generation) and fills content above it using only this track's own rolloff slope, harmonics, and dynamics - no external reference.", info: "tool_spectral_revive" },
    { id: "high_pass", group: "chainGroupCleanup", name: "High-pass filter", desc: "Removes inaudible sub-30Hz rumble that eats into headroom.", info: "tool_high_pass" },
    { id: "linear_fix", group: "chainGroupAI", name: "Linear model fix", desc: "Gradient-optimized correction targeting the fakeprint logistic-regression detector.", info: "linear_passes" },
    { id: "cnn_fix", group: "chainGroupAI", name: "CNN model fix", desc: "Shift-robust optimization targeting the CQT-cepstrum CNN detector. Slower.", info: "cnn_passes" },
    { id: "temporal_normalize", group: "chainGroupAI", name: "Temporal pattern denormalization", desc: "Applies a small, smooth, non-uniform timing drift, displacing the low-frequency spectral peaks that fingerprint matching uses as anchors. Off by default.", info: "temporal_normalize" },
    { id: "fix_phase", group: "chainGroupMaster", name: "Stereo phase correction", desc: "Corrects out-of-phase content that would cancel out in mono playback.", info: "tool_fix_phase" },
    { id: "normalize_lufs", group: "chainGroupMaster", name: "LUFS loudness normalization", desc: "Targets -14 LUFS, the standard streaming-platform loudness reference.", info: "lufs" },
    { id: "multiband_compress", group: "chainGroupMaster", name: "Multiband compression", desc: "Gentle 3-band dynamics smoothing for tonal balance. Repeats itself until the imbalance settles, so a peaky track is handled in one run.", info: "tool_multiband_compress" },
    { id: "true_peak_limit", group: "chainGroupMaster", name: "True-peak limiter", desc: "Brick-wall safety ceiling at -1dBTP, accounting for inter-sample peaks.", info: "tool_true_peak_limit" },
    { id: "fade", group: "chainGroupMaster", name: "Fade in / fade out", desc: "Smooth S-curve fade at the start and end of the track. Runs last, after the limiter, so no later gain stage undoes it.", info: "tool_fade" },
  ];

  // BUG FIX (second adversarial audit round): the analysis-response race
  // guard (runAnalysis's requestedFileId check) can only ever be as
  // correct as state.fileId itself - but the UPLOAD race was never
  // guarded at all. If upload A starts, then upload B starts, and B's
  // /api/upload response returns before A's, state.fileId briefly becomes
  // B's id (correct) - but if A's response then lands AFTER B's, A
  // unconditionally overwrites state.fileId back to A's older id with no
  // check at all, "winning" despite being the stale request. This counter
  // makes upload ordering explicit: each handleFile call captures the
  // sequence number in effect when IT started, and only applies its
  // result if no newer upload has started since.
  let uploadSequence = 0;
  // Declared up here rather than next to pollJob because clearProcessingLog
  // (called from clearAnalysisDisplay, far above pollJob) assigns it; a `let`
  // sits in the temporal dead zone until its declaration executes, so
  // leaving it below its first assignment site is a latent ReferenceError
  // the moment anything clears the log during initial page setup.
  let seenLogCount = 0;
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
    cnnMode: "thorough",
    // 4ms: fingerprint landmark displacement saturates here (landmark timing
    // is quantized by the 512-sample STFT hop, ~11.6ms, so once a landmark
    // moves past one hop it cannot move "further"). Higher values add high-
    // frequency smearing without adding disruption.
    temporalMaxDriftMs: 4,
    // 10ms in by default: trim_silence already removes the leading silence,
    // so this only needs to be long enough to avoid a click at the very
    // first sample. 3000ms out is a conventional musical fade.
    fadeInMs: 10,
    fadeOutMs: 3000,
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

  // cnnModeSwitch/temporalDriftSlider are rendered dynamically inside
  // renderToolChain() now (attached directly under their own tool's card,
  // not as page-load-time fixed elements) - listeners for them are wired
  // via delegation in renderToolChain() itself, right alongside the
  // tool-row click handler, since both need re-attaching every render.

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
    if (typeof stopElapsedTimer === "function") stopElapsedTimer();
    // BUG FIX (third adversarial audit round): resetWorkspace cleared
    // state.fileId but never incremented uploadSequence - so an upload
    // already in flight when the user clicked "Analyze a different file"
    // would still pass handleFile's own sequence guard (mySequence still
    // equals uploadSequence, since nothing bumped it here) and reopen the
    // just-discarded workspace with that stale upload's data the moment
    // its response landed. Bumping the sequence here invalidates any
    // upload started before this reset, the same way starting a genuinely
    // NEWER upload already does.
    uploadSequence++;
    // BUG FIX (direct user report): clicking "Analyze a different file" left
    // whatever was currently playing (the original-file preview player OR
    // the A/B before/after player) running in the background with no
    // indication anything was still audible - the workspace visually reset
    // to a blank upload state while audio kept going. The A/B play button's
    // own handler already stops the OTHER player when one starts (see
    // originalPlayer.onplay above) - this applies that same "only one
    // thing plays at a time" rule to the reset action itself.
    const origPlayer = $("originalPlayer");
    if (origPlayer && !origPlayer.paused) origPlayer.pause();
    const ao = $("audioOrig"), af = $("audioFixed");
    if (ao && !ao.paused) ao.pause();
    if (af && !af.paused) af.pause();
    document.querySelectorAll("#correctionOverlayList audio").forEach(a => {
      a.pause();
      a.removeAttribute("src");
      a.load();
    });
    $("correctionOverlayPanel")?.classList.add("hidden");
    const playBtn = $("playBtn");
    if (playBtn) playBtn.textContent = "▶";
    state = { ...state, fileId: null, analysis: null, selected: new Set(), jobId: null, result: null,
              lastResult: null, spectrumViewMode: "both", waveformViewMode: "both" };
    $("workspace").classList.remove("active");
    $("results").classList.remove("active");
    $("detectorAnalysisPanel").classList.remove("hidden");
    // Hiding the progress panel alone leaves the previous run's lines inside
    // it, which flash back the moment the next run re-shows the panel.
    clearProcessingLog();
    $("spectrumLegend").classList.remove("active");
    $("waveformLegend").classList.remove("active");
    $("analyzingState").classList.remove("active");
    document.querySelectorAll(".chart-view-row").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".chart-view-switch button").forEach(b => b.classList.toggle("active", b.dataset.view === "both"));
    fileInput.value = "";
  }

  async function handleFile(file) {
    const mySequence = ++uploadSequence;
    uploadZone.classList.add("dragover");
    uploadZone.querySelector(".primary").textContent = "Uploading & decoding…";
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      if (mySequence !== uploadSequence) {
        // a newer upload started (and possibly already finished) while
        // this one's request was in flight - this response is stale no
        // matter when it happens to arrive; discard it rather than let it
        // clobber whatever the truly latest upload already set.
        return;
      }
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
      // playing the original-file player should stop the A/B player, and
      // vice versa (wired on the A/B side's own play handler) - otherwise
      // both can play simultaneously with no indication anything's wrong
      origPlayer.onplay = () => {
        const ao = $("audioOrig"), af = $("audioFixed");
        if (ao && !ao.paused) ao.pause();
        if (af && !af.paused) af.pause();
        const playBtn = $("playBtn");
        if (playBtn) playBtn.textContent = "▶";
      };

      const stem = data.filename.replace(/\.[^.]+$/, "");
      $("outputNameInput").value = `${stem}_fixed.wav`;

      $("workspace").classList.add("active");
      $("results").classList.remove("active");
      $("detectorAnalysisPanel").classList.remove("hidden");
      renderToolChain();
      await runAnalysis();
    } catch (err) {
      if (mySequence === uploadSequence) {
        uploadZone.querySelector(".primary").textContent = "Upload failed — try again";
      }
      console.error(err);
    }
  }

  function fmtFadeMs(ms) {
    // sub-second values read better in ms; past that, seconds
    return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(ms % 1000 === 0 ? 0 : 1)}s`;
  }

  function fmtDuration(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  // ---------- analysis ----------
  // BUG FIX (direct user report): the Detector Analysis panel (score
  // cards, stat list, spectrum/waveform charts) kept showing the PREVIOUS
  // file's numbers while a new file's analysis was still in flight - the
  // "Analyzing audio..." banner rendered ABOVE that stale content in
  // normal document flow, not as an overlay covering it, so both were
  // simultaneously visible and looked like the new file was already
  // scored. Clear every field this function populates back to a neutral
  // "—" placeholder state immediately when analysis starts, before the
  // fetch even goes out, so nothing carries over between files.
  function clearAnalysisDisplay() {
    $("linearScore").textContent = "—";
    $("linearScore").className = "value mono";
    $("linearMeter").style.width = "0%";
    $("linearMeter").className = "meter-fill";
    $("linearPill").innerHTML = "";
    $("cnnScore").textContent = "—";
    $("cnnScore").className = "value mono";
    $("cnnMeter").style.width = "0%";
    $("cnnMeter").className = "meter-fill";
    $("cnnPill").innerHTML = "";
    $("statList").innerHTML = "";
    const specCtx = $("spectrumCanvas").getContext("2d");
    specCtx.clearRect(0, 0, $("spectrumCanvas").width, $("spectrumCanvas").height);
    const waveCtx = $("waveformOverviewCanvas").getContext("2d");
    waveCtx.clearRect(0, 0, $("waveformOverviewCanvas").width, $("waveformOverviewCanvas").height);
    $("waveformDuration").textContent = "";
    $("spectrumLegend").classList.remove("active");
    $("waveformLegend").classList.remove("active");
    // The Processing log belongs to whichever file was last RUN, so leaving
    // it on screen after uploading or re-analyzing a different file shows a
    // log that no longer describes the audio in the workspace. Clear it on
    // the same path that clears the Detector Analysis panel.
    clearProcessingLog();
  }

  function stopAllPlayback() {
    // Every <audio> on the page: the upload preview, the A/B pair, and the
    // correction-overlay players. Starting a new job must silence all of
    // them - otherwise the previous run's output keeps playing over the top
    // of a job that is busy replacing it.
    document.querySelectorAll("audio").forEach(a => {
      if (!a.paused) a.pause();
    });
    const playBtn = $("playBtn");
    if (playBtn) playBtn.textContent = "▶";
  }

  function clearResultCharts() {
    // Wipe everything the PREVIOUS run's results rendered, so clicking
    // "Process file" again does not leave the old run's spectrum, waveform
    // and correction overlays on screen - stale charts that describe an
    // output which no longer exists while the new job is still running.
    //
    // Deliberately NOT clearAnalysisDisplay(): the Detector Analysis panel
    // describes the uploaded FILE, which has not changed on a re-run, so
    // blanking its scores would throw away still-valid information. Only the
    // result-side charts are reset here.
    const specCtx = $("spectrumCanvas").getContext("2d");
    specCtx.clearRect(0, 0, $("spectrumCanvas").width, $("spectrumCanvas").height);
    const waveCtx = $("waveformOverviewCanvas").getContext("2d");
    waveCtx.clearRect(0, 0, $("waveformOverviewCanvas").width, $("waveformOverviewCanvas").height);
    $("waveformDuration").textContent = "";
    $("spectrumLegend").classList.remove("active");
    $("spectrumLegend").innerHTML = "";
    $("waveformLegend").classList.remove("active");
    $("waveformLegend").innerHTML = "";
    document.querySelectorAll(".chart-view-row").forEach(el => el.classList.remove("active"));
    // stop and detach the previous run's overlay players; leaving them live
    // means audio from the OLD output can keep playing over the new job.
    document.querySelectorAll("#correctionOverlayList audio").forEach(a => {
      a.pause();
      a.removeAttribute("src");
      a.load();
    });
    $("correctionOverlayPanel")?.classList.add("hidden");
    state.lastResult = null;
  }

  function clearProcessingLog() {
    $("logBox").innerHTML = "";
    $("progressPanel").classList.remove("active");
    $("progressFill").style.width = "0%";
    $("elapsedTime").textContent = "";
    // seenLogCount is pollJob's cursor into the job's log array and is NOT
    // derived from the DOM - blanking logBox without resetting it makes the
    // next job slice its short log against a stale larger cursor and drop
    // every early line (the same bug the third adversarial audit round found
    // on the runBtn path).
    seenLogCount = 0;
  }

  // BUG FIX (adversarial audit): runAnalysis had no way to tell, once its
  // fetch resolved, whether the user had since uploaded a NEWER file while
  // this (older, possibly slower) analysis was still in flight - if
  // analysis A returns after a newer analysis B already updated the
  // display, A's stale results silently overwrite B's current ones even
  // though state.fileId correctly shows B's id the whole time. Capture the
  // fileId THIS call is analyzing and only apply its response if that's
  // still the current file when the response lands; otherwise discard it
  // quietly - a real, if rare, race (rapid re-upload, or a slow analysis
  // on a large file followed immediately by a smaller/faster one).
  async function runAnalysis() {
    const requestedFileId = state.fileId;
    clearAnalysisDisplay();
    $("analyzingState").classList.add("active");
    const startedAt = performance.now();
    const tick = setInterval(() => {
      const secs = Math.round((performance.now() - startedAt) / 1000);
      $("analyzingLabel").textContent = secs < 3
        ? "Analyzing audio (running AI detectors, spectral checks)…"
        : `Analyzing audio (running AI detectors, spectral checks)… ${secs}s`;
    }, 1000);
    try {
      const res = await fetch(`/api/analyze/${requestedFileId}`);
      const data = await res.json();
      if (state.fileId !== requestedFileId) {
        // a newer upload superseded this one while the fetch was in
        // flight - discard these now-stale results instead of clobbering
        // whatever the current file's own analysis already rendered (or
        // will render, if its own runAnalysis call is still in flight too).
        return;
      }
      state.analysis = data;
      renderAnalysis(data);
      state.selected = new Set(data.recommended_tools);
      renderToolChain();
    } finally {
      // the interval timer belongs ONLY to this call and must always be
      // cleared regardless of staleness (leaving it running would leak a
      // timer forever) - but the shared "analyzing..." banner should only
      // be hidden by whichever call is actually still current, so a
      // slower stale request finishing after a newer one doesn't
      // incorrectly clear the spinner out from under a still-in-flight
      // current analysis.
      clearInterval(tick);
      if (state.fileId === requestedFileId) {
        $("analyzingState").classList.remove("active");
      }
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
        // BUG FIX (adversarial audit): this used a wider "good" band
        // (-16 to -12) than the backend's own /api/analyze recommendation
        // logic (only -17..-13 OR -15..-15 count as fine, everything else
        // recommends normalize_lufs) - confirmed directly a value like
        // -12.5 was recommended for correction by the backend but marked
        // "good" here. Match the backend's exact acceptance range.
        k: "LUFS (integrated)", v: `${data.lufs.toFixed(1)} LU`, info: "lufs",
        status: ((data.lufs >= -17 && data.lufs <= -13)) ? "good" : (data.lufs < -20 || data.lufs > -8) ? "warn" : null,
      },
      {
        k: "Stereo correlation", v: data.stereo_correlation.toFixed(2), info: "stereo_correlation",
        status: data.stereo_correlation >= 0.3 ? "good" : data.stereo_correlation < 0.1 ? "bad" : "warn",
      },
      {
        // BUG FIX (adversarial audit): this used a completely different
        // bar (0.001) than the backend's own /api/analyze recommendation
        // threshold (6e-5, DC_OFFSET_RECHECK_FLOOR in server.py) -
        // confirmed directly a value like 0.0005 was recommended for
        // correction by the backend but shown as "Safe" here. Match the
        // backend's exact threshold so the two never contradict each
        // other on the same page again.
        k: "DC offset (L/R)", v: `${data.dc_offset.l.toFixed(5)} / ${data.dc_offset.r.toFixed(5)}`, info: "dc_offset",
        status: dcMax > 6e-5 ? "warn" : "good",
      },
      {
        // BUG FIX (direct user report): this fired on ANY nonzero value,
        // while the backend's own /api/analyze recommendation logic uses a
        // 100ms bar (deliberately set there to tolerate genuinely
        // inaudible processing residue from resample/interpolation
        // round-trips elsewhere in the pipeline) - so this panel could
        // show a "Check" pill for a value (e.g. 30.5ms/69ms) the backend
        // itself had ALREADY decided needs no action and wouldn't even
        // recommend trim_silence for. Same 100ms bar here so the two
        // don't contradict each other on the same page.
        k: "Leading/trailing silence", v: `${data.silence.lead_ms || 0}ms / ${data.silence.trail_ms || 0}ms`, info: "silence_trim",
        status: ((data.silence.lead_ms || 0) > 100 || (data.silence.trail_ms || 0) > 100) ? "warn" : "good",
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
    drawWaveformOverview($("waveformOverviewCanvas"), data.waveform, null, "both", data.transients);
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

    // -100dB was too narrow a floor - confirmed directly on a real
    // production spectrum that legitimate content (both the source's own
    // measured noise floor AND the spectral-revival fill-in, which is
    // deliberately anchored just above that same floor) regularly sits at
    // -130 to -150dB up near 20-22kHz. Anything below the old -100dB floor
    // was silently clamped flat to the chart's bottom edge, which visually
    // read as "the after-curve rolls off/dies earlier than before" even
    // though the underlying data showed the fix correctly adding +23dB of
    // real content in that exact region - the chart was lying, not the fix.
    const minDb = -150, maxDb = 20;
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
        for (const m of markers) {
          const x = (m.time_sec / durationSec) * w;

          // vertical guide line through the whole waveform
          ctx.strokeStyle = crit;
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.5;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, h);
          ctx.stroke();
          ctx.globalAlpha = 1;

          // flag marker at the top
          ctx.fillStyle = crit;
          ctx.beginPath();
          ctx.moveTo(x - 4, 0);
          ctx.lineTo(x + 4, 0);
          ctx.lineTo(x, 7);
          ctx.closePath();
          ctx.fill();

          // always-visible timestamp label so the marker is self-explanatory
          // without requiring a hover - the legend explains WHAT the flag
          // means, this shows WHERE/WHEN
          const label = fmtDuration(m.time_sec);
          ctx.font = "9px ui-monospace, monospace";
          const labelWidth = ctx.measureText(label).width;
          const labelX = Math.min(Math.max(x - labelWidth / 2, 2), w - labelWidth - 2);
          ctx.fillStyle = crit;
          ctx.globalAlpha = 0.12;
          ctx.fillRect(labelX - 2, 9, labelWidth + 4, 11);
          ctx.globalAlpha = 1;
          ctx.fillStyle = crit;
          ctx.textBaseline = "top";
          ctx.fillText(label, labelX, 10);
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
        let settingsRow = "";
        if (t.id === "cnn_fix" && checked) {
          settingsRow = `
          <div class="tool-settings-row">
            <div class="chain-group-label">CNN fix mode<button class="info-btn" data-info="cnn_mode" title="What's this?" type="button">i</button></div>
            <div class="format-switch cols-3" id="cnnModeSwitch">
              <button data-cnnmode="simple" class="${state.cnnMode === "simple" ? "active" : ""}">Simple</button>
              <button data-cnnmode="eot" class="${state.cnnMode === "eot" ? "active" : ""}">Shift-robust</button>
              <button data-cnnmode="thorough" class="${state.cnnMode === "thorough" ? "active" : ""}">Thorough</button>
            </div>
            <div class="cnn-mode-hint" id="cnnModeHint">Recommended - trains the fix to hold up even if the check lands slightly off-position.</div>
          </div>`;
        } else if (t.id === "temporal_normalize" && checked) {
          settingsRow = `
          <div class="tool-settings-row">
            <div class="chain-group-label">Max timing drift</div>
            <div class="slider-row">
              <input type="range" id="temporalDriftSlider" min="2" max="15" step="1" value="${state.temporalMaxDriftMs}">
              <span class="slider-value mono" id="temporalDriftValue">${state.temporalMaxDriftMs}ms</span>
            </div>
            <div class="cnn-mode-hint">Fingerprint matching keys on low-frequency peaks — 94% below 500Hz, none above 4kHz. Displacement saturates at 4ms, so higher values only smear sibilants ("s"/"t" sounds) without adding disruption.</div>
          </div>`;
        } else if (t.id === "fade" && checked) {
          settingsRow = `
          <div class="tool-settings-row">
            <div class="chain-group-label">Fade in</div>
            <div class="slider-row">
              <input type="range" id="fadeInSlider" min="10" max="10000" step="10" value="${state.fadeInMs}">
              <span class="slider-value mono" id="fadeInValue">${fmtFadeMs(state.fadeInMs)}</span>
            </div>
            <div class="chain-group-label">Fade out</div>
            <div class="slider-row">
              <input type="range" id="fadeOutSlider" min="10" max="10000" step="10" value="${state.fadeOutMs}">
              <span class="slider-value mono" id="fadeOutValue">${fmtFadeMs(state.fadeOutMs)}</span>
            </div>
            <div class="cnn-mode-hint">Smooth S-curve fade. Applied after the limiter so nothing later undoes it.</div>
          </div>`;
        }
        // Show WHAT this file's imbalance is, so the recommendation has a
        // reason attached. It used to end with "may take another pass or two
        // to fully clear", which read as an instruction to re-upload and
        // re-run by hand - the tool now iterates internally until the file's
        // measured peakiness stops improving, so there is nothing for the
        // user to repeat.
        let peakinessHint = "";
        if (t.id === "multiband_compress" && recommended && state.analysis && state.analysis.band_peakiness) {
          const worstBand = state.analysis.band_peakiness.reduce(
            (a, b) => (b.peak_over_db > a.peak_over_db ? b : a)
          );
          if (worstBand.peak_over_db > 0) {
            peakinessHint = `<div class="cnn-mode-hint">${worstBand.peak_over_db.toFixed(1)}dB over target in the ${worstBand.range_hz[0]}-${worstBand.range_hz[1]}Hz band. Runs as many gentle passes as it takes to settle.</div>`;
          }
        }
        return `
        <div class="tool-row ${checked ? "checked" : ""}" data-tool="${t.id}">
          <div class="box"></div>
          <div class="info">
            <div class="name">${t.name}${t.info ? infoBtn(t.info) : ""}${recommended ? `<span class="rec-badge">Recommended</span>` : ""}</div>
            <div class="desc">${t.desc}</div>
            ${peakinessHint}
            ${settingsRow}
          </div>
        </div>`;
      }).join("");
    }

    document.querySelectorAll(".tool-row").forEach(row => {
      row.addEventListener("click", (e) => {
        // clicking the info (i) button, or interacting with a tool's own
        // inline settings row (CNN mode switch, temporal drift slider),
        // should never also toggle the tool's checkbox - this listener is
        // attached directly on the row (fires before the global document-
        // level info-btn handler even sees the bubbled event), so
        // stopPropagation inside that later handler is too late to prevent
        // this one from already running. Bail out here directly instead.
        if (e.target.closest(".info-btn") || e.target.closest(".tool-settings-row")) return;
        const id = row.dataset.tool;
        if (state.selected.has(id)) state.selected.delete(id);
        else state.selected.add(id);
        renderToolChain();
      });
    });

    // cnnModeSwitch/temporalDriftSlider are re-created every render (they
    // only exist inside their own tool's card while that tool is checked),
    // so their listeners must be re-attached every render too, not once at
    // page load.
    const cnnModeSwitchEl = document.getElementById("cnnModeSwitch");
    if (cnnModeSwitchEl) {
      cnnModeSwitchEl.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          state.cnnMode = btn.dataset.cnnmode;
          cnnModeSwitchEl.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === btn));
        });
      });
    }
    const temporalSliderEl = document.getElementById("temporalDriftSlider");
    if (temporalSliderEl) {
      temporalSliderEl.addEventListener("input", (e) => {
        state.temporalMaxDriftMs = Number(e.target.value);
        document.getElementById("temporalDriftValue").textContent = `${state.temporalMaxDriftMs}ms`;
      });
      temporalSliderEl.addEventListener("click", (e) => e.stopPropagation());
    }
    for (const [sliderId, valueId, key] of [
      ["fadeInSlider", "fadeInValue", "fadeInMs"],
      ["fadeOutSlider", "fadeOutValue", "fadeOutMs"],
    ]) {
      const el = document.getElementById(sliderId);
      if (!el) continue;
      el.addEventListener("input", (e) => {
        state[key] = Number(e.target.value);
        document.getElementById(valueId).textContent = fmtFadeMs(state[key]);
      });
      // without this the click bubbles to the tool row and toggles the tool
      // off mid-drag, the same reason the drift slider stops propagation
      el.addEventListener("click", (e) => e.stopPropagation());
    }

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

  $("copyLogBtn").addEventListener("click", async () => {
    const lines = Array.from($("logBox").querySelectorAll(".line")).map(
      line => line.dataset.raw || line.textContent
    );
    const text = lines.join("\n");
    const btn = $("copyLogBtn");
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
      btn.classList.add("copied");
    } catch (err) {
      btn.textContent = "Copy failed";
    }
    setTimeout(() => {
      btn.textContent = "Copy log";
      btn.classList.remove("copied");
    }, 1500);
  });

  // ---------- run pipeline ----------
  let elapsedTimer = null;
  function startElapsedTimer() {
    const startedAt = performance.now();
    stopElapsedTimer();
    elapsedTimer = setInterval(() => {
      const secs = Math.floor((performance.now() - startedAt) / 1000);
      const m = Math.floor(secs / 60);
      const s = secs % 60;
      $("elapsedTime").textContent = `${m}:${String(s).padStart(2, "0")} elapsed`;
    }, 1000);
  }
  function stopElapsedTimer() {
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
  }

  $("runBtn").addEventListener("click", async () => {
    if (!state.fileId || state.selected.size === 0) return;
    // clearProcessingLog resets the visible log AND seenLogCount together
    // (pollJob's cursor into the job's log array): blanking one without the
    // other makes the next job slice its short log against a stale larger
    // cursor and silently drop every early line - the bug the third
    // adversarial audit round found here. It also removes .active, so it
    // must run BEFORE the panel is shown below.
    // Silence anything currently playing before the new job starts - the
    // audio being played is about to be replaced by this run's output.
    stopAllPlayback();
    clearProcessingLog();
    // Reset the previous run's result charts too - hiding #results leaves the
    // old spectrum/waveform pixels and legends intact underneath, so they
    // reappear unchanged the moment the new run finishes rendering, and are
    // visibly stale if the user re-opens the panel mid-job.
    clearResultCharts();
    $("progressPanel").classList.add("active");
    $("results").classList.remove("active");
    $("progressFill").style.width = "6%";
    $("runBtn").disabled = true;
    $("cancelJobBtn").classList.remove("hidden");
    $("elapsedTime").textContent = "0:00 elapsed";
    startElapsedTimer();

    const outputName = $("outputNameInput").value.trim();
    const res = await fetch(`/api/process/${state.fileId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tools: Array.from(state.selected), options: { cnn_mode: state.cnnMode, temporal_max_drift_ms: state.temporalMaxDriftMs, fade_in_ms: state.fadeInMs, fade_out_ms: state.fadeOutMs },
        output_name: outputName || undefined,
        output_format: state.outputFormat,
        mp3_mode: state.mp3Mode,
      }),
    });
    const data = await res.json();
    if (data.error) {
      appendLog(data.error, true);
      $("runBtn").disabled = false;
      $("cancelJobBtn").classList.add("hidden");
      return;
    }
    state.jobId = data.job_id;
    pollJob();
  });

  $("cancelJobBtn").addEventListener("click", async () => {
    if (!state.jobId) return;
    $("cancelJobBtn").disabled = true;
    try {
      const res = await fetch(`/api/job/${state.jobId}/cancel`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);
      appendLog("Cancelling… optimizers now check between iterations, but a real-model scoring pass already in progress may still take a little while.");
    } catch (err) {
      appendLog(`Could not cancel: ${err}`, true);
    } finally {
      $("cancelJobBtn").disabled = false;
    }
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

    // BUG FIX (direct user report + screenshot): post-chain reverify passes
    // (a CNN or linear model getting re-run because a later stage disturbed
    // it) run entirely OUTSIDE the numbered 13-tool chain - the backend now
    // marks this by sending current_step_idx/total_steps as null while
    // still setting current_step_name and sub_progress. This used to fall
    // straight into the "job just started, no step info yet" branch below,
    // which threw away sub_progress entirely and froze the progress bar at
    // a flat 4% - so a multi-minute CNN retry showed a dead progress bar
    // and (before current_step_name existed for this case) the stale
    // heading/count from whichever tool ran last in the real chain (e.g.
    // "Tool 13 of 13 (True-peak limiter)" while a CNN re-verification pass
    // was actually running for several more minutes). Handle it as its own
    // case: show the real reverify label with no misleading "Tool N of N"
    // wrapper, and still render sub_progress so the optimization-step
    // counter and live score estimate keep working exactly as they do
    // inside the normal chain.
    if (idx == null && data.current_step_name) {
      $("progressStepLabel").textContent = data.current_step_name;
      let subText = "";
      if (data.sub_progress && data.sub_progress.total) {
        const sp = data.sub_progress;
        const fracWithinStep = Math.min(1, sp.current / sp.total);
        $("progressFill").style.width = `${Math.min(99, Math.round(fracWithinStep * 100))}%`;
        subText = `step ${sp.current} of ${sp.total}`;
        if (sp.attempt && sp.max_attempts) subText += ` · attempt ${sp.attempt}/${sp.max_attempts}`;
        if (sp.score_pct !== undefined) subText += ` · live estimate ${sp.score_pct}% AI`;
        if (sp.windows_total !== undefined) {
          const passing = sp.windows_total - sp.windows_failing;
          subText += ` · real-detector check: ${passing}/${sp.windows_total} spots passing (worst spot ${sp.real_max_score_pct}% AI)`;
        }
      } else {
        $("progressFill").style.width = "99%";
      }
      $("progressStepCount").textContent = subText;
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

    // Two genuinely different counters were being smashed into one run-on
    // sentence: idx/total is which TOOL in the selected chain is running
    // (e.g. tool 12 of 12 selected tools), while sub_progress is that one
    // tool's OWN internal optimization loop (e.g. gradient step 90 of 300)
    // - a user reported this as unreadable ("this makes no sense"), and
    // reading it back it's easy to see why: "Step 12 of 12 (optimization
    // step 90 of 300...)" looks like one counter contradicting itself, not
    // two different things. Render them as two separate lines instead.
    // BUG FIX (direct user report): "Tool 10 of 13" alone doesn't say
    // WHICH tool is running - during a long CNN/linear optimization the
    // user is watching "Optimization step 2 of 488" with no indication in
    // that same line whether this is the linear or CNN model being fixed.
    // current_step_name already holds the real tool label (rendered
    // separately in progressStepLabel above) - fold it in here too so the
    // step-count line is self-sufficient on its own.
    let stepText = `Tool ${idx + 1} of ${total}`;
    if (data.current_step_name) stepText += ` (${data.current_step_name})`;
    let subText = "";
    if (data.sub_progress && data.sub_progress.total) {
      const sp = data.sub_progress;
      subText = `Optimization step ${sp.current} of ${sp.total}`;
      if (sp.attempt && sp.max_attempts) subText += ` · attempt ${sp.attempt}/${sp.max_attempts}`;
      if (sp.score_pct !== undefined) subText += ` · live estimate ${sp.score_pct}% AI`;
      if (sp.windows_total !== undefined) {
        // "windows" is internal jargon (the optimizer's overlapping analysis
        // segments, not anything the user selected or can see) - rephrase
        // in terms of what the user actually cares about: how close is the
        // real detector check to fully passing.
        const passing = sp.windows_total - sp.windows_failing;
        subText += ` · real-detector check: ${passing}/${sp.windows_total} spots passing (worst spot ${sp.real_max_score_pct}% AI)`;
      }
    }
    $("progressStepCount").innerHTML = subText
      ? `${stepText}<br><span class="sub-progress-detail">${subText}</span>`
      : stepText;
  }

  // translates the technical pipeline log into a plain-language line for
  // the primary display, while keeping the raw text available (small,
  // dimmed, underneath) for anyone who wants the exact detail - rather
  // than rewriting the raw strings themselves, which stay precise for
  // debugging.
  const LOG_TRANSLATIONS = [
    [/^loading .+$/, () => "Loading your file…"],
    [/^running: wm$/, () => "Applying product watermark"],
    [/^running: (.+)$/, (m) => `Starting: ${m[1]}`],
    [/^\s*done \(([\d.]+)s\)$/, (m) => `Done (${m[1]}s)`],
    [/^wm: pass \(version (\d+), (\d+)% confidence, method=(\w+)\)$/, (m) =>
      ({ text: `Watermark (wm): embedded and verified — ${m[2]}% confidence`, badge: "pass" })],
    [/^wm: fail.*$/, () => ({ text: "Watermark (wm): embedded but could not be re-verified", badge: "fail" })],
    [/^wm: error.*$/, () => ({ text: "Watermark (wm): step failed, file shipped without it", badge: "fail" })],
    [/^linear: attempt (\d+) of (\d+) - optimizing.*$/, (m) => `AI-detector fix (linear model): trying attempt ${m[1]} of ${m[2]}…`],
    [/^linear: attempt \d+ result checked against the REAL detector.*: ([\d.]+)%$/, (m) => {
      const pct = Number(m[1]);
      return { text: `Checked against the real detector: ${m[1]}% AI-likely`, badge: pct < 1 ? "pass" : "retry" };
    }],
    [/^linear: real score [\d.]+% is above the <1% target - retrying.*$/, () => ({ text: "Not quite under 1% yet - trying again with a stricter setting", badge: "retry" })],
    [/^linear: real score [\d.]+% is above the <1% target after all \d+ attempts.*$/, () => ({ text: "Didn't fully clear 1% after all attempts - shipping the best result found", badge: "fail" })],
    [/^cnn: optimizing ([\d.]+)s of audio.*$/, (m) => `AI-detector fix (CNN model): working through ${Math.round(m[1])}s of audio - this one takes a while…`],
    [/^cnn: verifying final transferred stereo output against the real model$/, () => "Double-checking the CNN fix on the actual file you'll receive"],
    [/^cnn: worst window after transfer scored ([\d.]+)% AI$/, (m) => {
      const pct = Number(m[1]);
      return { text: `Checked against the real detector: worst window ${m[1]}% AI-likely`, badge: pct < 8 ? "margin" : "retry" };
    }],
    [/^cnn: no windows survived the post-transfer check.*$/, () => ({ text: "Could not verify any window after transfer - track may be too short", badge: "fail" })],
    [/^re-verifying linear model after full chain.*$/, () => "Double-checking the linear fix still holds after mastering"],
    [/^\s*post-chain linear score: ([\d.]+)%$/, (m) => {
      const pct = Number(m[1]);
      return { text: `Linear model result: ${m[1]}% AI-likely`, badge: pct < 1 ? "pass" : "retry" };
    }],
    [/^\s*above target - re-running linear_fix.*$/, () => ({ text: "Still above target - running the linear fix once more", badge: "retry" })],
    [/^\s*post-chain cnn score: ([\d.]+)%$/, (m) => {
      const pct = Number(m[1]);
      return { text: `CNN model result: ${m[1]}% AI-likely`, badge: pct < 8 ? "margin" : "retry" };
    }],
    [/^\s*cnn lost its safety margin.*$/, () => ({ text: "The CNN fix slipped after a later step - running it again", badge: "retry" })],
    [/^temporal_normalize: note .*$/, () => "Temporal denormalization stayed in place; a later safety correction ran before the final watermark (this exact combination is unbenchmarked)"],
    [/^\s*found (\d+) new anomal(?:y|ies) introduced by later chain stages.*$/, (m) => ({
      text: `Transient/pop fix: found ${m[1]} new anomal${m[1] === "1" ? "y" : "ies"} from later steps - fixing`,
      badge: "retry",
    })],
    [/^fix_transients: final (pass|check) \((\d+) anomal(?:y|ies) after full chain\)$/, (m) => ({
      text: `Transient/pop fix: ${m[2]} anomal${m[2] === "1" ? "y" : "ies"} remain after the full chain`,
      badge: m[1] === "pass" ? "pass" : "retry",
    })],
    // Generic handler for the centralized per-tool status line (see
    // _tool_status_line in server.py) - every selected tool logs exactly
    // one "  {tool}: pass|check (...)" line right after "done (Xs)", so one
    // regex + a friendly-name lookup covers all of them instead of a
    // separate hand-written pattern per tool (which is exactly how several
    // tools ended up with no status line at all before this).
    [/^\s*(strip_metadata|trim_silence|dc_offset|fix_transients|spectral_revive|high_pass|fix_phase|normalize_lufs|multiband_compress|temporal_normalize|true_peak_limit|fade): (pass|check) \((.+)\)$/, (m) => {
      const FRIENDLY_TOOL_NAME = {
        strip_metadata: "Metadata strip", trim_silence: "Silence trim", dc_offset: "DC offset correction",
        fix_transients: "Transient/pop fix", spectral_revive: "High-frequency fill-in", high_pass: "High-pass filter",
        fix_phase: "Stereo phase correction", normalize_lufs: "Loudness normalization",
        multiband_compress: "Multiband compression", temporal_normalize: "Temporal pattern denormalization",
        true_peak_limit: "True-peak limiter", fade: "Fade in / fade out",
      };
      const name = FRIENDLY_TOOL_NAME[m[1]] || m[1];
      return { text: `${name}: ${m[3]}`, badge: m[2] === "pass" ? "pass" : "retry" };
    }],
    [/^re-running true-peak limiter.*$/, () => "Re-checking the loudness ceiling after that last change"],
    [/^post-chain LUFS check: ([\-\d.]+) vs target ([\-\d.]+).*$/, (m) => `Loudness drifted to ${m[1]} (target ${m[2]}) - correcting`],
    [/^\s*corrected to ([\-\d.]+) LUFS.*$/, (m) => `Loudness corrected to ${m[1]}`],
    [/^saving output file.*$/, () => "Saving your finished file…"],
    [/^re-scoring with AI detectors$/, () => "Running the final check with both AI detectors"],
    [/^WARNING: final file still flagged by at least one model \((.+)\)$/, (m) => ({ text: `Heads up: still flagged by at least one detector (${m[1]})`, badge: "fail" })],
    [/^\s*WARNING: linear regressed.*$/, () => ({ text: "Heads up: the linear score slipped a bit after a later step", badge: "fail" })],
    [/^\s*WARNING: cnn regressed.*$/, () => ({ text: "Heads up: the CNN score slipped again after the loudness limiter re-ran - the delivered file may still be flagged", badge: "fail" })],
    [/^\s*WARNING: delivered file is ([\-\d.]+) LUFS.*$/, (m) => `Heads up: couldn't fully reach the loudness target (landed at ${m[1]} LUFS) without exceeding the peak safety ceiling`],
    // Keep the FULL tag list verbatim - it is the actual evidence of what was
    // found in the file, not a summary. Only a badge is added.
    [/^\s*found and removing tags: (.+)$/, (m) => ({ text: `found and removing tags: ${m[1]}`, badge: "pass" })],
    [/^\s*no text tags found on the source file$/, () => ({ text: "no text tags found on the source file", badge: "pass" })],
    [/^\s*found and removing (\d+) embedded image\(s\)(.*)$/, (m) => ({
      text: `found and removing ${m[1]} embedded image(s)${m[2]}`, badge: "pass" })],
    [/^\s*no embedded images found$/, () => ({ text: "no embedded images found", badge: "pass" })],
    [/^linear: trying fast feature-domain solve.*$/, () => "Trying the fast solve before the full optimizer"],
    [/^linear: feature-domain result checked on transferred stereo output: ([\d.]+)% AI, SNR ([\d.]+)dB, peak spectral adjustment ([\d.]+)dB$/, (m) => {
      const pct = Number(m[1]);
      return { text: `Fast solve checked on the real output: ${m[1]}% AI-likely, SNR ${m[2]}dB, peak EQ change ${m[3]}dB`,
               badge: pct < 1 ? "pass" : "retry" };
    }],
    [/^\s*skipped (\d+) pre-existing anomal(?:y|ies) already present in the source.*$/, (m) => ({
      text: `Transient/pop fix: left ${m[1]} pre-existing anomal${m[1] === "1" ? "y" : "ies"} alone (already in your source, not added by this chain)`,
      badge: "pass",
    })],
    [/^total processing time: (.+)$/, (m) => ({ text: `Total processing time: ${m[1]}`, badge: "pass" })],
    [/^complete$/, () => "All done"],
    [/^cancelled by user$/, () => ({ text: "Job cancelled", badge: "fail" })],
    [/^ERROR: (.+)$/, (m) => `Something went wrong: ${m[1]}`],
  ];

  function friendlyLog(msg) {
    for (const [pattern, fn] of LOG_TRANSLATIONS) {
      const m = msg.match(pattern);
      if (m) return fn(m);
    }
    return null;
  }

  const LOG_BADGE_LABEL = { pass: "PASS", retry: "RETRY", fail: "FLAGGED", margin: "WITHIN TARGET" };

  function appendLog(msg, isErr) {
    const box = $("logBox");
    const line = document.createElement("div");
    line.className = "line" + (isErr ? " err" : "");
    const result = friendlyLog(msg);
    const friendlyText = result && typeof result === "object" ? result.text : result;
    const badge = result && typeof result === "object" ? result.badge : null;
    if (friendlyText && friendlyText !== msg) {
      // friendly translation only - no raw line underneath. Showing both
      // (as an earlier version did) doubled the visual volume of every
      // single log event across a job that can run 20+ minutes and emit
      // hundreds of lines - confirmed too noisy in practice, not just in
      // theory. Raw text is still in data-raw for anyone who wants to
      // inspect it (e.g. via devtools), just not rendered by default.
      line.className += " line-main";
      line.dataset.raw = msg;
      const textSpan = document.createElement("span");
      textSpan.textContent = friendlyText;
      line.appendChild(textSpan);
      if (badge) {
        const badgeSpan = document.createElement("span");
        badgeSpan.className = `log-badge ${badge}`;
        badgeSpan.textContent = LOG_BADGE_LABEL[badge] || badge;
        line.appendChild(badgeSpan);
      }
    } else {
      line.textContent = msg;
    }
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  // BUG FIX (third adversarial audit round): pollJob captured state.jobId
  // only implicitly (reading it fresh each recursive setTimeout call), but
  // a single in-flight fetch has no record of WHICH job it was requested
  // for - if the user cancels and starts a new job while an old poll's
  // request is still in flight, that stale response lands with no way to
  // tell it apart from the current job's own response, and would
  // overwrite seenLogCount/progress/results with data for a job that's no
  // longer the one being watched.
  async function pollJob() {
    if (!state.jobId) return;
    const polledJobId = state.jobId;
    try {
      const res = await fetch(`/api/job/${polledJobId}`);
      const data = await res.json();
      if (state.jobId !== polledJobId) {
        // a newer job started while this request was in flight - this
        // response describes a job that's no longer current; discard it
        // rather than let it clobber the actually-current job's state.
        return;
      }

      const newLines = data.log.slice(seenLogCount);
      seenLogCount = data.log.length;
      for (const l of newLines) appendLog(l.msg);

      updateProgress(data);

      if (data.status === "running") {
        state.pollTimer = setTimeout(pollJob, 1200);
      } else if (data.status === "done") {
        stopElapsedTimer();
        $("runBtn").disabled = false;
        $("cancelJobBtn").classList.add("hidden");
        state.result = data.result;
        renderResults(data.result);
      } else if (data.status === "error") {
        stopElapsedTimer();
        appendLog(`Failed: ${data.error}`, true);
        $("runBtn").disabled = false;
        $("cancelJobBtn").classList.add("hidden");
      } else if (data.status === "cancelled") {
        stopElapsedTimer();
        appendLog("Job cancelled.", true);
        $("runBtn").disabled = false;
        $("cancelJobBtn").classList.add("hidden");
      }
    } catch (err) {
      appendLog(String(err), true);
      state.pollTimer = setTimeout(pollJob, 2000);
    }
  }

  // ---------- results ----------
  function renderResults(result) {
    $("results").classList.add("active");
    // BUG FIX (direct user report): nothing in the pre-processing grid
    // (Detector Analysis, the Processing log, Signal Chain) gets hidden
    // when results render - the user explicitly wants all of it to stay on
    // screen alongside the results panel, not disappear.

    const passBanner = $("verdictBanner");
    $("verdictFilename").textContent = state.filename || "";
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

    const dcMaxBefore = result.dc_offset_before ? Math.max(Math.abs(result.dc_offset_before.l), Math.abs(result.dc_offset_before.r)) : null;
    const dcMaxAfter = result.dc_offset_after ? Math.max(Math.abs(result.dc_offset_after.l), Math.abs(result.dc_offset_after.r)) : null;
    const rows = [
      ["Linear model score", `${result.scores_before.linear_pct.toFixed(3)}%`, `${result.scores_after.linear_pct.toFixed(3)}%`, result.scores_after.passes_linear, "linear_passes"],
      ["CNN model score", `${result.scores_before.cnn_pct.toFixed(1)}%`, `${result.scores_after.cnn_pct.toFixed(1)}%`, result.scores_after.passes_cnn, "cnn_passes"],
      // BUG FIX (second adversarial audit round): this used a WIDER
      // "good" band (-16..-12) than /api/analyze's own recommendation
      // logic (-17..-13) - confirmed directly a delivered file at -12.5
      // LUFS showed "good" here but got recommended for normalize_lufs
      // again the moment that same file was re-uploaded. The earlier fix
      // this session only synchronized the PRE-processing analysis
      // panel's threshold, reasoning the results table served a
      // different purpose (did the tool hit its own delivery bar) - that
      // reasoning doesn't survive contact with a user re-uploading their
      // own output and seeing a direct contradiction. Match exactly.
      ["Integrated LUFS", `${result.lufs_before.toFixed(1)}`, `${result.lufs_after.toFixed(1)}`,
       (result.lufs_after >= -17 && result.lufs_after <= -13) ? "good" : (result.lufs_after < -20 || result.lufs_after > -8) ? "bad" : "warn", "lufs"],
      ["Stereo correlation", result.stereo_correlation_before != null ? result.stereo_correlation_before.toFixed(2) : "—",
       result.stereo_correlation_after != null ? result.stereo_correlation_after.toFixed(2) : "—",
       result.stereo_correlation_after != null ? result.stereo_correlation_after >= 0.1 : null, "stereo_correlation"],
      // BUG FIX (second adversarial audit round): same class of gap as
      // LUFS above - this used 0.001 while /api/analyze's own
      // recommendation floor is 6e-5 (DC_OFFSET_RECHECK_FLOOR in
      // server.py, itself calibrated against real measured MP3 encoder
      // noise). Confirmed directly a delivered value of 0.0005 passed
      // here but triggered a dc_offset recommendation on re-upload.
      ["DC offset (max L/R)", dcMaxBefore != null ? dcMaxBefore.toFixed(5) : "—", dcMaxAfter != null ? dcMaxAfter.toFixed(5) : "—",
       dcMaxAfter != null ? dcMaxAfter < 6e-5 : null, "dc_offset"],
      ["Transients detected", result.transients_found ? (result.transients_found.length + (result.transients_found.length ? " (fixed)" : "")) : "—",
       result.transients_after_count != null ? result.transients_after_count : "—",
       result.transients_after_count != null ? result.transients_after_count === 0 : null, "transients"],
      ["Duration", fmtDuration(result.duration_sec), fmtDuration(result.duration_sec), null, null],
      ["Signal-to-noise (original vs. fixed)", "n/a (reference)", result.overall_snr_db ? `${result.overall_snr_db.toFixed(1)} dB` : "unchanged", null, "snr"],
    ];
    if (result.spectrum_before && result.spectrum_before.tilt && result.spectrum_after && result.spectrum_after.tilt) {
      const tb = result.spectrum_before.tilt, ta = result.spectrum_after.tilt;
      const fmtTilt = t => `${t["low (20-250Hz)"].toFixed(0)} / ${t["mid (250-4000Hz)"].toFixed(0)} / ${t["high (4000-20000Hz)"].toFixed(0)} dB`;
      rows.push(["Spectral tilt (low/mid/high)", fmtTilt(tb), fmtTilt(ta), null, "spectral_tilt"]);
    }
    function statusPill(status) {
      // supports two shapes: boolean (existing pass/fail rows - linear,
      // cnn, stereo correlation, dc offset, transients) and a "good"/
      // "warn"/"bad" string (LUFS, which has a real middle ground - close
      // to target but not exact isn't a hard fail, unlike a boolean would
      // force it to render as).
      if (status === null || status === undefined) return "—";
      if (status === true) return '<span class="pill good">pass</span>';
      if (status === false) return '<span class="pill crit">flagged</span>';
      if (status === "good") return '<span class="pill good">pass</span>';
      if (status === "warn") return '<span class="pill warn">check</span>';
      if (status === "bad") return '<span class="pill crit">flagged</span>';
      return "—";
    }
    $("compareTable").innerHTML = `
      <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Status</th></tr></thead>
      <tbody>
      ${rows.map(([k, before, after, pass, info]) => `
        <tr>
          <td>${k}${info ? infoBtn(info) : ""}</td><td>${before}</td><td>${after}</td>
          <td>${statusPill(pass)}</td>
        </tr>`).join("")}
      </tbody>`;

    $("stepsList").innerHTML = result.steps.map(s => {
      const warning = stepWarningText(s, result);
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
      const hasTransientMarkers = result.transients_found && result.transients_found.length > 0;
      $("waveformLegend").innerHTML = `
        <div class="item"><span class="swatch" style="background: var(--before)"></span>Before</div>
        <div class="item"><span class="swatch" style="background: var(--accent)"></span>After</div>
        ${hasTransientMarkers ? `<div class="item"><span class="swatch" style="background: var(--crit)"></span>Fixed transient/pop${result.transients_found.length > 1 ? ` (${result.transients_found.length})` : ""} — timestamp shown at each marker</div>` : ""}
      `;
      document.querySelector("#waveformOverviewCanvas").closest(".spectrum-wrap").querySelector(".chart-view-row").classList.add("active");
    }

    setupABPlayer(result);
    renderCorrectionOverlays(result);
  }

  function renderCorrectionOverlays(result) {
    const panel = $("correctionOverlayPanel");
    const list = $("correctionOverlayList");
    const overlays = result.correction_overlays || {};
    const kinds = ["linear", "cnn", "combined"].filter(k => overlays[k]);
    if (!kinds.length) {
      list.innerHTML = "";
      panel.classList.add("hidden");
      return;
    }
    const labels = {
      linear: "Linear correction",
      cnn: "CNN correction",
      combined: "Combined",
    };
    list.innerHTML = kinds.map(kind => {
      const gain = Number(overlays[kind].preview_gain_db || 0);
      return `
        <div class="overlay-row">
          <div class="overlay-label">${labels[kind]}</div>
          <div>
            <span class="overlay-player-label">Actual level</span>
            <audio controls preload="none" src="/api/audio/overlay_${kind}/${result.out_id}"></audio>
          </div>
          <div>
            <span class="overlay-player-label">Amplified preview (+${gain.toFixed(1)} dB)</span>
            <audio controls preload="none" src="/api/audio/overlay_${kind}_loud/${result.out_id}"></audio>
          </div>
        </div>`;
    }).join("");
    list.querySelectorAll("audio").forEach(player => {
      player.addEventListener("play", () => {
        $("audioOrig").pause();
        $("audioFixed").pause();
        list.querySelectorAll("audio").forEach(other => {
          if (other !== player) other.pause();
        });
      });
    });
    panel.classList.remove("hidden");
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
    drawWaveformOverview($("waveformOverviewCanvas"), result.waveform_before, result.waveform_after, mode, result.transients_found);
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
    else if (s.tool === "cnn_fix" || s.tool === "cnn_fix_reverify") {
      text = ` — SNR ${s.snr_db.toFixed(1)}dB`;
      if (s.worst_score_after_transfer !== undefined && s.worst_score_after_transfer !== null) {
        // This is a deliberately pessimistic ROBUSTNESS check, not the real
        // detector's actual verdict: it scans a small window around each of
        // the 5 real evaluation spots and reports the worst score found
        // ANYWHERE nearby, as a safety margin against the file shifting
        // slightly. The real detector's own verdict (checked later, after
        // the rest of the chain, at its own exact positions) is what
        // actually determines pass/fail - see the CNN model score row
        // below and the verdict banner above, not this number.
        text += `, safety-margin check (worst nearby spot): ${(s.worst_score_after_transfer * 100).toFixed(1)}% AI`;
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

  function stepWarningText(s, result) {
    // Surface the honesty-signal fields that mean "this fix didn't fully
    // reach its target" - these are set specifically so a partial fix is
    // never silently reported as a full success.
    if (s.warning) return s.warning;
    if ((s.tool === "cnn_fix" || s.tool === "cnn_fix_reverify") && s.verified_after_transfer === false) {
      // verified_after_transfer is keyed to 0.08 - a deliberately tighter
      // safety margin than the real detector's own 0.5 pass/fail boundary,
      // used so the optimizer keeps pushing for comfortable headroom rather
      // than a knife's-edge pass. That means this CAN legitimately read
      // "not verified" here while the real detector, checked later against
      // its own actual positions (median of 5, not worst-of-nearby), still
      // passes comfortably - confirmed happening in practice (worst nearby
      // spot 8.3%, real final CNN score 0.1%). Only surface this as a
      // warning if the actual final verdict agrees something is wrong;
      // otherwise this step earned real safety margin even if not the full
      // margin it was aiming for, and flagging it here just contradicts the
      // pass shown in the verdict banner a few inches away for no reason.
      const finalCnnPct = result && result.scores_after ? result.scores_after.cnn_pct : null;
      if (finalCnnPct === null || finalCnnPct >= 50) {
        return `Worst nearby spot still scored ${(s.worst_score_after_transfer * 100).toFixed(1)}% AI after transfer, and the final detector check agrees - this file may still get flagged.`;
      }
      return null;
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
        // starting A/B playback should stop the original-file player at
        // the top of the page too - otherwise both can play at once
        const origPlayer = $("originalPlayer");
        if (origPlayer && !origPlayer.paused) origPlayer.pause();
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
