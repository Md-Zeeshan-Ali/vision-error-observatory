/* Vision Error Observatory: interactive site.
   Everything shown is computed from data/app_data_v2.js (real NYU Depth V2 test scenes) or copied
   from the committed results summarised in the repository README. */
(() => {
  "use strict";

  // ---------- constants ----------
  const { W, H, CROP, clamp, ease, smooth, mkCanvas, TURBO, INFERNO, INFERNO_GLSL, lutCss } = window.VEO;
  const { fx: FX, fy: FY, cx: CX, cy: CY } = window.VEO.CAM;
  const LN2 = Math.log(2);

  const LABELS = {
    boundary_mixing: { name: "Depth boundary", color: "#ff7a45" },
    thin_object: { name: "Thin object", color: "#ffd166" },
    reflection: { name: "Reflection", color: "#59d0ff" },
    textureless: { name: "Textureless", color: "#b892ff" },
    poor_lighting: { name: "Poor lighting", color: "#7ee2a8" },
    ambiguous: { name: "Ambiguous", color: "#8d909c" },
  };
  const LABEL_ORDER = ["boundary_mixing", "thin_object", "reflection", "textureless", "poor_lighting"];

  const LAYERS = [
    { id: "photo", name: "Photo" },
    { id: "truth", name: "Sensor depth" },
    { id: "pred", name: "Prediction" },
    { id: "error", name: "Error" },
    { id: "fail", name: "Failing" },
  ];

  // README: "Do the labels work?" (AUROC against annotations, 0.5 is chance)
  const LABEL_RESULTS = [
    { id: "boundary_mixing", au: [0.85, 0.81, 0.69], verdict: ["Holds up", "good"], note: "Weaker on iBims-1, which uses annotated depth edges." },
    { id: "thin_object", au: [0.73, 0.81, null], verdict: ["Holds up where testable", "good"], note: "iBims-1 has no thin-object annotation." },
    { id: "reflection", au: [0.83, 0.39, 0.82], verdict: ["Sensor-dependent", "mixed"], note: "Rides on sensor holes, which alone score 0.83 on iBims-1." },
    { id: "textureless", au: [0.58, 0.50, null], verdict: ["Close to chance", "weak"], note: "Only 3 iBims-1 regions fall on its wall masks." },
    { id: "poor_lighting", au: null, verdict: ["Not validated", "none"], note: "A hand-set rule: no annotation exists to fit or score it against." },
    { id: "ambiguous", au: null, verdict: ["Abstains", "none"], note: "When no label reaches its threshold, VEO declines to name a cause." },
  ];
  const AU_SETS = ["NYU test", "Hypersim", "iBims-1"];

  // README: "Do the explanations hold when the model is re-run?"
  const INTERVENTIONS = [
    { edit: "Higher resolution", label: "boundary_mixing", est: -0.009, lo: -0.016, hi: -0.004, p: "0.003", ok: true, tag: "Supported (exploratory)" },
    { edit: "Higher resolution", label: "thin_object", est: -0.004, lo: -0.011, hi: 0.003, p: "0.63", ok: false, tag: "Not supported" },
    { edit: "Brighten", label: "poor_lighting", est: -0.007, lo: -0.020, hi: 0.007, p: "0.72", ok: false, tag: "Not supported" },
    { edit: "Add texture", label: "textureless", est: 0.014, lo: -0.019, hi: 0.050, p: "0.72", ok: false, tag: "Not supported" },
  ];

  // README: "How do models compare?"
  const MODELS = [
    { name: "Depth Anything V2 Large", v: [0.043, 0.979, 2.1, 0.065, 0.038] },
    { name: "Depth Pro", v: [0.043, 0.979, 2.1, 0.064, 0.037] },
    { name: "Depth Anything V2 Base", v: [0.046, 0.977, 2.3, 0.068, 0.041] },
    { name: "ZoeDepth NYU+KITTI*", v: [0.050, 0.976, 2.4, 0.089, 0.066] },
    { name: "Depth Anything V2 Small", v: [0.051, 0.973, 2.7, 0.073, 0.047], me: true },
    { name: "Marigold†", v: [0.059, 0.959, 4.1, 0.086, 0.053] },
    { name: "DPT-Hybrid", v: [0.091, 0.918, 8.2, 0.098, 0.075] },
    { name: "DPT-Large", v: [0.092, 0.915, 8.5, 0.096, 0.071] },
  ];
  const METRICS = [
    { name: "NYU AbsRel", lower: true, fmt: (v) => v.toFixed(3) },
    { name: "NYU δ₁", lower: false, fmt: (v) => v.toFixed(3) },
    { name: "NYU failing px", lower: true, fmt: (v) => v.toFixed(1) + "%" },
    { name: "DIODE AbsRel", lower: true, fmt: (v) => v.toFixed(3) },
    { name: "iBims-1 AbsRel", lower: true, fmt: (v) => v.toFixed(3) },
  ];

  // ---------- helpers ----------
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const fmtInt = (n) => n >= 10000 ? (n / 1000).toFixed(0) + "k" : n.toLocaleString("en-US");
  const finePointer = matchMedia("(pointer: fine)").matches;
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let toastTimer;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 1800);
  }

  // ---------- data loading ----------
  function loadData() {
    return window.VEO.loadData((f) => {
      $("#loader-fill").style.width = pct(f, 0);
      $("#loader-pct").textContent = Math.round(f * 100) + "%";
    });
  }

  // ---------- scene preparation ----------
  async function prepScene(sc) {
    const P = await window.VEO.prepScene(sc);
    if (P.layers) return P;
    P.layers = {};
    const { rgb, gt, pd, lr, crop, lo, hi } = P;
    const mk = () => { const cv = mkCanvas(W, H); const x = cv.getContext("2d"); return { cv, x, d: x.createImageData(W, H) }; };
    const L = { photo: mk(), truth: mk(), pred: mk(), error: mk() };
    const span = hi - lo || 1;
    const ph = L.photo.d.data, tr = L.truth.d.data, pr = L.pred.d.data, er = L.error.d.data;
    const put = (dst, o, r, g, b) => { dst[o] = r; dst[o + 1] = g; dst[o + 2] = b; dst[o + 3] = 255; };
    for (let i = 0; i < W * H; i++) {
      const o = i * 4, r = rgb[o], g = rgb[o + 1], b = rgb[o + 2];
      if (!crop[i]) {
        const gray = (r * 0.3 + g * 0.59 + b * 0.11) * 0.25;
        put(ph, o, r * 0.35, g * 0.35, b * 0.35);
        put(tr, o, gray, gray, gray); put(pr, o, gray, gray, gray); put(er, o, gray, gray, gray);
        continue;
      }
      put(ph, o, r, g, b);
      let k = Math.round(clamp((gt[i] - lo) / span) * 255) * 3;
      put(tr, o, TURBO[k], TURBO[k + 1], TURBO[k + 2]);
      k = Math.round(clamp((pd[i] - lo) / span) * 255) * 3;
      put(pr, o, TURBO[k], TURBO[k + 1], TURBO[k + 2]);
      k = Math.round(clamp(lr[i] / LN2) * 255) * 3;
      put(er, o, INFERNO[k], INFERNO[k + 1], INFERNO[k + 2]);
    }
    for (const k in L) { L[k].x.putImageData(L[k].d, 0, 0); P.layers[k] = L[k].cv; }
    P.failLayer = mk();
    P.layers.fail = P.failLayer.cv;
    return P;
  }

  // Recolour the "failing" layer for a threshold; returns the failing fraction of crop pixels.
  function paintFail(P, thr) {
    const lt = Math.log(thr);
    const { d, x } = P.failLayer;
    const data = d.data, rgb = P.rgb;
    let n = 0, f = 0;
    for (let i = 0; i < W * H; i++) {
      const o = i * 4;
      const gray = (rgb[o] * 0.3 + rgb[o + 1] * 0.59 + rgb[o + 2] * 0.11);
      if (!P.crop[i]) { const v = gray * 0.2; data[o] = data[o + 1] = data[o + 2] = v; data[o + 3] = 255; continue; }
      n++;
      if (P.lr[i] >= lt) {
        f++;
        const s = clamp((P.lr[i] - lt) / 0.45);
        const k = Math.round((0.55 + 0.45 * s) * 255) * 3;
        const a = 0.75;
        data[o] = INFERNO[k] * a + gray * 0.25 * (1 - a);
        data[o + 1] = INFERNO[k + 1] * a + gray * 0.25 * (1 - a);
        data[o + 2] = INFERNO[k + 2] * a + gray * 0.25 * (1 - a);
      } else {
        const v = gray * 0.42;
        data[o] = v; data[o + 1] = v * 1.02; data[o + 2] = v * 1.08;
      }
      data[o + 3] = 255;
    }
    x.putImageData(d, 0, 0);
    return n ? f / n : 0;
  }

  // ---------- point cloud (three.js) ----------
  const VERT = `
    uniform float uMix, uTime, uColorMode, uThr, uGlow, uScan, uScanOn, uBoxOn, uProj, uStep, uSizeMul;
    uniform vec4 uBox;
    attribute vec3 aPD;
    attribute vec3 aCol;
    attribute float aErr;
    attribute vec2 aPix;
    attribute float aRnd;
    varying vec3 vCol;
    ${INFERNO_GLSL}
    void main() {
      float m = clamp(uMix * 1.5 - aRnd * 0.5, 0.0, 1.0);
      m = m * m * (3.0 - 2.0 * m);
      vec3 pos = mix(position, aPD, m);
      vec3 col = uColorMode > 0.5 ? inferno(clamp(aErr / 0.6931, 0.0, 1.0) * 0.94 + 0.03) : aCol;
      float fail = step(uThr, aErr);
      float flick = 0.6 + 0.4 * sin(uTime * 3.0 + aRnd * 40.0);
      col = mix(col, vec3(1.0, 0.35, 0.21), fail * uGlow * 0.5 * flick);
      float d = -pos.z;
      float scan = uScanOn * exp(-pow((d - uScan) * 6.0, 2.0));
      col += scan * vec3(0.30, 0.78, 1.0) * 0.9;
      float inBox = uBoxOn * step(uBox.x, aPix.x) * step(aPix.x, uBox.x + uBox.z) * step(uBox.y, aPix.y) * step(aPix.y, uBox.y + uBox.w);
      float pulse = 0.5 + 0.5 * sin(uTime * 5.0);
      col = mix(col, vec3(1.0), inBox * (0.2 + 0.35 * pulse));
      vec4 mv = modelViewMatrix * vec4(pos, 1.0);
      gl_Position = projectionMatrix * mv;
      float world = uStep * d / ${FX.toFixed(4)};
      gl_PointSize = max(1.0, world * uProj / -mv.z * uSizeMul * (1.0 + inBox * 0.5 + fail * uGlow * 0.2));
      vCol = col;
    }`;
  const FRAG = `
    varying vec3 vCol;
    void main() {
      vec2 c = gl_PointCoord - 0.5;
      if (dot(c, c) > 0.25) discard;
      gl_FragColor = vec4(vCol, 1.0);
    }`;

  class PointCloud {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.opts = opts;
      this.pr = Math.min(window.devicePixelRatio || 1, 2);
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
      this.renderer.setPixelRatio(this.pr);
      this.renderer.setClearColor(0x000000, 0);
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(opts.fov || 55, 1, 0.01, 60);
      this.step = opts.step || 2;
      this.uniforms = {
        uMix: { value: 0 }, uTime: { value: 0 }, uColorMode: { value: 0 }, uThr: { value: Math.log(1.25) },
        uGlow: { value: 1 }, uScan: { value: -10 }, uScanOn: { value: 1 }, uBoxOn: { value: 0 },
        uBox: { value: new THREE.Vector4() }, uProj: { value: 500 }, uStep: { value: this.step }, uSizeMul: { value: opts.sizeMul || 1.15 },
      };
      this.material = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: VERT, fragmentShader: FRAG });
      this.target = new THREE.Vector3(0, 0, -2.5);
      this.visible = true;
      this.onFrame = null;
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; }, { rootMargin: "100px" }).observe(canvas);
      this.resize();
      const loop = (t) => {
        requestAnimationFrame(loop);
        if (!this.visible || !this.points) return;
        this.uniforms.uTime.value = t / 1000;
        if (this.onFrame) this.onFrame(t / 1000);
        this.renderer.render(this.scene, this.camera);
      };
      requestAnimationFrame(loop);
    }
    resize() {
      const r = this.canvas.parentElement.getBoundingClientRect();
      const w = Math.max(1, r.width), h = Math.max(1, r.height);
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      if (this.opts.offsetX && w > 900) this.camera.setViewOffset(w, h, -w * this.opts.offsetX, 0, w, h);
      else if (this.camera.view) this.camera.clearViewOffset();
      this.camera.updateProjectionMatrix();
      this.uniforms.uProj.value = (h * this.pr) / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    }
    setScene(P) {
      const s = this.step;
      const n = Math.ceil((CROP.x1 - CROP.x0) / s) * Math.ceil((CROP.y1 - CROP.y0) / s);
      const gp = new Float32Array(n * 3), pp = new Float32Array(n * 3), col = new Float32Array(n * 3);
      const err = new Float32Array(n), pix = new Float32Array(n * 2), rnd = new Float32Array(n);
      let k = 0, zs = 0;
      for (let y = CROP.y0; y < CROP.y1; y += s) {
        for (let x = CROP.x0; x < CROP.x1; x += s) {
          const i = y * W + x;
          if (!P.crop[i]) continue;
          const g = P.gt[i], p = P.pd[i];
          gp[k * 3] = ((x - CX) * g) / FX; gp[k * 3 + 1] = (-(y - CY) * g) / FY; gp[k * 3 + 2] = -g;
          pp[k * 3] = ((x - CX) * p) / FX; pp[k * 3 + 1] = (-(y - CY) * p) / FY; pp[k * 3 + 2] = -p;
          const o = i * 4;
          col[k * 3] = P.rgb[o] / 255; col[k * 3 + 1] = P.rgb[o + 1] / 255; col[k * 3 + 2] = P.rgb[o + 2] / 255;
          err[k] = P.lr[i];
          pix[k * 2] = x; pix[k * 2 + 1] = y;
          rnd[k] = ((x - CROP.x0) / (CROP.x1 - CROP.x0)) * 0.75 + Math.random() * 0.25;
          zs += g;
          k++;
        }
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(gp.subarray(0, k * 3), 3));
      geo.setAttribute("aPD", new THREE.BufferAttribute(pp.subarray(0, k * 3), 3));
      geo.setAttribute("aCol", new THREE.BufferAttribute(col.subarray(0, k * 3), 3));
      geo.setAttribute("aErr", new THREE.BufferAttribute(err.subarray(0, k), 1));
      geo.setAttribute("aPix", new THREE.BufferAttribute(pix.subarray(0, k * 2), 2));
      geo.setAttribute("aRnd", new THREE.BufferAttribute(rnd.subarray(0, k), 1));
      if (this.points) { this.scene.remove(this.points); this.points.geometry.dispose(); }
      this.points = new THREE.Points(geo, this.material);
      this.points.frustumCulled = false;
      this.scene.add(this.points);
      const zMean = zs / Math.max(1, k);
      this.target.set(0, -0.05, -zMean);
      this.zRange = [P.lo, P.hi];
      if (this.opts.onScene) this.opts.onScene(this);
    }
    setBox(r) {
      if (!r) { this.uniforms.uBoxOn.value = 0; return; }
      const [x, y, w, h] = r.bbox;
      this.uniforms.uBox.value.set(x, y, w, h);
      this.uniforms.uBoxOn.value = 1;
    }
  }

  // ---------- app state ----------
  const state = {
    data: null, scenes: [], tax: {},
    sc: null, P: null, region: null, hover: null,
    thr: 1.25, mode: "wipe", A: "photo", B: "error",
    wipeX: W * 0.5, pointer: null, filter: new Set([...LABEL_ORDER, "ambiguous"]),
    hero: null, lift: null,
  };

  const regionColor = (r) => LABELS[r.labels.length ? r.top_cause : "ambiguous"].color;
  const regionKey = (r) => (r.labels.length ? r.top_cause : "ambiguous");
  const focusLabel = (sc) => ((sc.default_region.desc || "").match(/labelled (\w+)/) || [])[1];
  const visibleRegions = () => state.sc.regions.filter((r) => state.filter.has(regionKey(r)));

  function defaultRegion(sc) {
    const { x, y } = sc.default_region;
    const lab = focusLabel(sc);
    const hit = sc.regions.filter((r) => r.labels.includes(lab) && x >= r.bbox[0] && x <= r.bbox[0] + r.bbox[2] && y >= r.bbox[1] && y <= r.bbox[1] + r.bbox[3]);
    // skip regions whose box spans most of the crop: they are scattered and read as a frame, not a region
    const cropArea = (CROP.x1 - CROP.x0) * (CROP.y1 - CROP.y0);
    const compact = (r) => r.bbox[2] * r.bbox[3] < 0.6 * cropArea;
    const labelled = sc.regions.filter((r) => r.labels.includes(lab));
    const pool = [hit.filter(compact), labelled.filter(compact), hit, labelled, sc.regions].find((p) => p.length);
    return pool.slice().sort((a, b) => b.area - a.area)[0];
  }

  // ---------- explorer ----------
  const view = { cv: null, cx: null, dpr: 1, dirty: true, hintGone: false };

  function initExplorer() {
    view.cv = $("#view-canvas");
    view.dpr = Math.min(window.devicePixelRatio || 1, 2);
    view.cv.width = W * view.dpr;
    view.cv.height = H * view.dpr;
    view.cx = view.cv.getContext("2d");

    const strip = $("#scene-strip");
    state.scenes.forEach((sc, idx) => {
      const b = el("button", "scene-tab");
      b.setAttribute("role", "tab");
      b.innerHTML = `<img alt="" src="${sc.images.rgb}"><span class="st-badge">AbsRel ${sc.metrics.abs_rel.toFixed(3)}</span>
        <div class="st-body"><div class="st-id">${sc.sample_id} · ${sc.regions.length} regions</div><div class="st-title">${sc.title}</div></div>`;
      b.onclick = () => selectScene(idx);
      strip.appendChild(b);
    });

    const chipRow = (host, key) => {
      LAYERS.forEach((L) => {
        const c = el("button", "chip", L.name);
        c.dataset.layer = L.id;
        c.onclick = () => { state[key] = L.id; syncLayerChips(); requestDraw(); };
        host.appendChild(c);
      });
    };
    chipRow($("#pick-a"), "A");
    chipRow($("#pick-b"), "B");
    syncLayerChips();

    $$("#explore .seg button[data-mode]").forEach((b) => {
      b.onclick = () => {
        state.mode = b.dataset.mode;
        $$("#explore .seg button[data-mode]").forEach((x) => x.classList.toggle("on", x === b));
        $("#pick-b-wrap").style.display = state.mode === "single" ? "none" : "";
        $("#viewer-hint").textContent = state.mode === "wipe" ? "drag the handle · click a box" : state.mode === "lens" ? "move the lens · click a box" : "hover for depth · click a box";
        syncLayerChips();
        requestDraw();
      };
    });

    const thr = $("#thr");
    thr.addEventListener("input", () => {
      state.thr = +thr.value;
      $("#thr-out").textContent = state.thr.toFixed(2);
      updateThreshold();
    });

    const v = $("#viewer");
    let drag = false, down = null;
    const toNative = (e) => {
      const r = view.cv.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * W, y: ((e.clientY - r.top) / r.height) * H };
    };
    v.addEventListener("pointerdown", (e) => {
      const p = toNative(e);
      down = { x: p.x, y: p.y };
      if (state.mode === "wipe" && (Math.abs(p.x - state.wipeX) < 22 || e.pointerType !== "mouse")) {
        drag = true;
        v.setPointerCapture(e.pointerId);
        if (e.pointerType !== "mouse") state.wipeX = clamp(p.x, 0, W);
      }
      hideHint();
      requestDraw();
    });
    v.addEventListener("pointermove", (e) => {
      const p = toNative(e);
      state.pointer = p;
      if (drag) state.wipeX = clamp(p.x, 0, W);
      v.style.cursor = state.mode === "wipe" && Math.abs(p.x - state.wipeX) < 22 ? "ew-resize" : state.mode === "lens" ? "none" : "crosshair";
      state.hover = regionAt(p.x, p.y);
      updateReadout();
      requestDraw();
    });
    const end = (e) => {
      const p = toNative(e);
      if (down && Math.hypot(p.x - down.x, p.y - down.y) < 6) {
        const r = regionAt(p.x, p.y);
        if (r) selectRegion(r);
      }
      drag = false;
      down = null;
    };
    v.addEventListener("pointerup", end);
    v.addEventListener("pointercancel", () => { drag = false; down = null; });
    v.addEventListener("pointerleave", () => {
      if (drag) return;
      state.pointer = null;
      state.hover = null;
      updateReadout();
      requestDraw();
    });

    window.addEventListener("keydown", (e) => {
      if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
      const r = $("#explore").getBoundingClientRect();
      if (r.bottom < 0 || r.top > innerHeight) return;
      if (e.key >= "1" && e.key <= "5") { state.A = LAYERS[+e.key - 1].id; syncLayerChips(); requestDraw(); }
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const list = railOrder();
        if (!list.length) return;
        let i = list.indexOf(state.region);
        i = (i + (e.key === "ArrowRight" ? 1 : -1) + list.length) % list.length;
        selectRegion(list[i]);
        e.preventDefault();
      }
    });

    let onscreen = true;
    new IntersectionObserver(([e]) => { onscreen = e.isIntersecting; }).observe(v);
    const loop = () => { requestAnimationFrame(loop); if (view.dirty && onscreen) { view.dirty = false; draw(); } };
    requestAnimationFrame(loop);
  }

  function hideHint() {
    if (view.hintGone) return;
    view.hintGone = true;
    $("#viewer-hint").style.opacity = 0;
  }

  function syncLayerChips() {
    $$("#pick-a .chip").forEach((c) => c.classList.toggle("on", c.dataset.layer === state.A));
    $$("#pick-b .chip").forEach((c) => c.classList.toggle("on", c.dataset.layer === state.B));
    const name = (id) => LAYERS.find((l) => l.id === id).name;
    $("#tag-a").textContent = name(state.A);
    $("#tag-b").textContent = state.mode === "lens" ? "Lens: " + name(state.B) : name(state.B);
    $("#tag-b").style.display = state.mode === "wipe" ? "" : "none";
  }

  const requestDraw = () => { view.dirty = true; };

  function regionAt(x, y) {
    let best = null;
    for (const r of visibleRegions()) {
      const [bx, by, bw, bh] = r.bbox;
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh && (!best || r.area < best.area)) best = r;
    }
    return best;
  }

  function draw() {
    const P = state.P;
    if (!P) return;
    const c = view.cx;
    c.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
    c.imageSmoothingEnabled = true;
    c.drawImage(P.layers[state.A], 0, 0, W, H);

    if (state.mode === "wipe") {
      const x = state.wipeX;
      c.save();
      c.beginPath(); c.rect(x, 0, W - x, H); c.clip();
      c.drawImage(P.layers[state.B], 0, 0, W, H);
      c.restore();
    }

    // eval crop frame
    c.save();
    c.setLineDash([4, 5]);
    c.strokeStyle = "rgba(255,255,255,.22)";
    c.lineWidth = 1;
    c.strokeRect(CROP.x0 + 0.5, CROP.y0 + 0.5, CROP.x1 - CROP.x0 - 1, CROP.y1 - CROP.y0 - 1);
    c.restore();

    // regions, largest first so small ones sit on top
    const regs = visibleRegions().slice().sort((a, b) => b.area - a.area);
    for (const r of regs) {
      if (r === state.region) continue;
      const [x, y, w, h] = r.bbox;
      c.strokeStyle = regionColor(r);
      c.globalAlpha = r === state.hover ? 1 : 0.55;
      c.lineWidth = r === state.hover ? 2 : 1;
      c.strokeRect(x + 0.5, y + 0.5, w, h);
    }
    c.globalAlpha = 1;
    if (state.region && state.filter.has(regionKey(state.region))) drawSelected(c, state.region);

    if (state.mode === "wipe") {
      const x = state.wipeX;
      c.fillStyle = "rgba(255,255,255,.95)";
      c.fillRect(x - 1, 0, 2, H);
      c.beginPath(); c.arc(x, H / 2, 15, 0, Math.PI * 2);
      c.fillStyle = "#eeece6"; c.fill();
      c.fillStyle = "#06070b";
      c.beginPath(); c.moveTo(x - 4, H / 2 - 5); c.lineTo(x - 9, H / 2); c.lineTo(x - 4, H / 2 + 5); c.fill();
      c.beginPath(); c.moveTo(x + 4, H / 2 - 5); c.lineTo(x + 9, H / 2); c.lineTo(x + 4, H / 2 + 5); c.fill();
    }

    const p = state.pointer;
    if (p && state.mode === "lens") {
      const R = 74, Z = 2;
      c.save();
      c.beginPath(); c.arc(p.x, p.y, R, 0, Math.PI * 2); c.clip();
      c.fillStyle = "#000"; c.fillRect(p.x - R, p.y - R, R * 2, R * 2);
      c.drawImage(P.layers[state.B], p.x - R / Z, p.y - R / Z, (R * 2) / Z, (R * 2) / Z, p.x - R, p.y - R, R * 2, R * 2);
      c.restore();
      c.beginPath(); c.arc(p.x, p.y, R, 0, Math.PI * 2);
      c.strokeStyle = "rgba(238,236,230,.95)"; c.lineWidth = 2; c.stroke();
      c.beginPath(); c.arc(p.x, p.y, R + 5, -Math.PI * 0.25, Math.PI * 0.25);
      c.strokeStyle = "rgba(255,90,54,.9)"; c.lineWidth = 2; c.stroke();
    }
    if (p) {
      c.strokeStyle = "rgba(255,255,255,.75)";
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(p.x - 8, p.y); c.lineTo(p.x - 3, p.y); c.moveTo(p.x + 3, p.y); c.lineTo(p.x + 8, p.y);
      c.moveTo(p.x, p.y - 8); c.lineTo(p.x, p.y - 3); c.moveTo(p.x, p.y + 3); c.lineTo(p.x, p.y + 8);
      c.stroke();
    }
  }

  function drawSelected(c, r) {
    const [x, y, w, h] = r.bbox;
    const col = regionColor(r);
    const t = performance.now() / 1000;
    c.save();
    c.shadowColor = col; c.shadowBlur = 12;
    c.strokeStyle = col; c.lineWidth = 2;
    c.setLineDash([6, 4]); c.lineDashOffset = -t * 20;
    c.strokeRect(x + 0.5, y + 0.5, w, h);
    c.setLineDash([]);
    const L = Math.max(6, Math.min(14, w / 3, h / 3));
    c.lineWidth = 3;
    c.beginPath();
    c.moveTo(x - 3, y - 3 + L); c.lineTo(x - 3, y - 3); c.lineTo(x - 3 + L, y - 3);
    c.moveTo(x + w + 3 - L, y - 3); c.lineTo(x + w + 3, y - 3); c.lineTo(x + w + 3, y - 3 + L);
    c.moveTo(x + w + 3, y + h + 3 - L); c.lineTo(x + w + 3, y + h + 3); c.lineTo(x + w + 3 - L, y + h + 3);
    c.moveTo(x - 3 + L, y + h + 3); c.lineTo(x - 3, y + h + 3); c.lineTo(x - 3, y + h + 3 - L);
    c.stroke();
    c.restore();
    const label = `#${r.region_id} ${LABELS[regionKey(r)].name}`;
    c.font = "600 11px 'JetBrains Mono', monospace";
    const tw = c.measureText(label).width + 12;
    let lx = clamp(x - 3, 2, W - tw - 2), ly = y - 24;
    if (ly < 2) ly = y + h + 6;
    c.fillStyle = col;
    c.beginPath();
    if (c.roundRect) c.roundRect(lx, ly, tw, 18, 9); else c.rect(lx, ly, tw, 18);
    c.fill();
    c.fillStyle = "#06070b";
    c.fillText(label, lx + 6, ly + 13);
    view.dirty = true; // keep the marching ants moving
  }

  function updateReadout() {
    const p = state.pointer, P = state.P;
    const set = (id, v, cls) => { const e = $(id); e.textContent = v; e.className = cls || ""; };
    if (!p || !P) {
      ["#ro-gt", "#ro-pd", "#ro-ratio", "#ro-verdict"].forEach((id) => set(id, "—"));
      $("#ro-px").textContent = "hover the image";
      return;
    }
    const x = Math.floor(p.x), y = Math.floor(p.y);
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const i = y * W + x;
    const g = P.gt[i], d = P.pd[i];
    set("#ro-gt", g.toFixed(2) + " m");
    set("#ro-pd", d.toFixed(2) + " m");
    const ratio = Math.max(d / g, g / d);
    set("#ro-ratio", "×" + ratio.toFixed(2));
    if (!P.crop[i]) {
      set("#ro-verdict", "outside", "muted");
      $("#ro-px").textContent = `(${x}, ${y}) · outside the evaluation crop`;
      return;
    }
    const fails = ratio >= state.thr;
    set("#ro-verdict", fails ? "fails" : "within δ", fails ? "verdict-bad" : "verdict-ok");
    const dir = d > g ? "too far" : "too near";
    const off = Math.abs(d - g) / g;
    $("#ro-px").textContent = `(${x}, ${y}) · prediction ${pct(off)} ${dir}` + (state.hover ? ` · region #${state.hover.region_id}` : "");
  }

  function updateThreshold() {
    if (!state.P) return;
    const f = paintFail(state.P, state.thr);
    $("#thr-frac").textContent = pct(f);
    const off = Math.abs(state.thr - 1.25) < 0.005;
    $("#thr-note").textContent = off ? `· VEO's measured-pixel figure: ${pct(state.sc.metrics.failure_pixel_fraction)}` : "· VEO uses 1.25";
    if (state.lift) state.lift.uniforms.uThr.value = Math.log(state.thr);
    if (state.hero) state.hero.uniforms.uThr.value = Math.log(state.thr);
    updateReadout();
    requestDraw();
  }

  function renderSceneMetrics() {
    const m = state.sc.metrics;
    const items = [
      ["AbsRel", m.abs_rel.toFixed(3)], ["RMSE", m.rmse.toFixed(3) + " m"],
      ["δ₁ < 1.25", pct(m.delta1)], ["Failing px", pct(m.failure_pixel_fraction)],
      ["Regions", m.total_regions], ["Spearman", m.spearman.toFixed(3)],
    ];
    $("#scene-metrics").innerHTML = items.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
    const P = state.P;
    $(".legend-bar").innerHTML = `<span class="mono muted">${P.lo.toFixed(1)} m</span><i style="background:${lutCss(TURBO)}"></i><span class="mono muted">${P.hi.toFixed(1)} m</span>
      <span class="mono muted" style="margin-left:14px">×1</span><i style="width:90px;background:${lutCss(INFERNO)}"></i><span class="mono muted">×2+</span>`;
  }

  function renderFilter() {
    const host = $("#label-filter");
    host.innerHTML = "";
    const present = new Set(state.sc.regions.map(regionKey));
    [...LABEL_ORDER, "ambiguous"].filter((k) => present.has(k)).forEach((k) => {
      const n = state.sc.regions.filter((r) => regionKey(r) === k).length;
      const c = el("button", "chip" + (state.filter.has(k) ? " on" : " off"), `<i class="d" style="background:${LABELS[k].color}"></i>${LABELS[k].name} <span class="muted">${n}</span>`);
      c.onclick = () => {
        state.filter.has(k) ? state.filter.delete(k) : state.filter.add(k);
        renderFilter();
        renderRail();
        requestDraw();
      };
      host.appendChild(c);
    });
  }

  const railOrder = () => visibleRegions().slice().sort((a, b) => b.mean_error - a.mean_error);

  function renderRail() {
    const host = $("#region-rail");
    host.innerHTML = "";
    railOrder().forEach((r) => {
      const k = regionKey(r);
      const b = el("button", "rail-item" + (r === state.region ? " on" : ""));
      b.innerHTML = `<div class="ri-top"><span>#${r.region_id}</span><span class="conf ${r.confidence}">${r.confidence}</span></div>
        <div class="ri-lab"><i class="d" style="width:7px;height:7px;border-radius:50%;background:${LABELS[k].color}"></i>${LABELS[k].name}${r.labels.length > 1 ? ` <span class="muted">+${r.labels.length - 1}</span>` : ""}</div>
        <div class="ri-err">AbsRel ${r.mean_error.toFixed(3)} · ${fmtInt(r.area)} px</div>`;
      b.onclick = () => selectRegion(r);
      b._r = r;
      host.appendChild(b);
    });
  }

  function selectRegion(r) {
    state.region = r;
    $$("#region-rail .rail-item").forEach((b) => {
      const on = b._r === r;
      b.classList.toggle("on", on);
      if (on) b.scrollIntoView({ block: "nearest", inline: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
    });
    renderDossier(r);
    if (state.lift) state.lift.setBox(r);
    if (state.hero) state.hero.setBox(r);
    const k = regionKey(r);
    $("#lift-region").innerHTML = `<span style="color:${LABELS[k].color}">●</span> Region #${r.region_id} · ${LABELS[k].name}`;
    requestDraw();
  }

  function renderDossier(r) {
    const k = regionKey(r);
    const T = state.tax[k] || {};
    const f = r.features || {};
    const tags = r.labels.length
      ? r.labels.map((l) => `<span class="tag" style="color:${LABELS[l].color};background:color-mix(in srgb, ${LABELS[l].color} 14%, transparent)"><i class="d" style="width:6px;height:6px;border-radius:50%;background:currentColor"></i>${LABELS[l].name}</span>`).join("")
      : `<span class="tag" style="color:var(--muted);background:var(--panel-2)">Abstains: no label reaches its threshold</span>`;
    const se = f.signed_error;
    const bias = se == null ? "—" : `${pct(Math.abs(se), 0)} ${se > 0 ? "too far" : "too near"}`;
    const scores = LABEL_ORDER.map((l) => {
      const v = r.scores[l] || 0;
      return `<div class="score"><span>${LABELS[l].name}</span><div class="bar"><i data-w="${v * 100}" style="background:${LABELS[l].color}"></i></div><span class="v">${v.toFixed(2)}</span></div>`;
    }).join("");
    $("#dossier-body").innerHTML = `
      <div class="dos-head"><div class="dos-title">Region #${r.region_id}</div><span class="conf ${r.confidence}">${r.confidence} confidence</span></div>
      <div class="tags">${tags}</div>
      <p class="dos-desc">${T.desc || ""}</p>
      <div class="dos-stats">
        <div><span>Mean AbsRel</span><b>${r.mean_error.toFixed(3)}</b></div>
        <div><span>Area</span><b>${r.area.toLocaleString("en-US")} px</b></div>
        <div><span>Depth</span><b>${f.mean_depth_m != null ? f.mean_depth_m.toFixed(2) + " m" : "—"}</b></div>
        <div><span>Predicted</span><b>${bias}</b></div>
        <div><span>Sensor holes</span><b>${f.sensor_hole_fraction != null ? pct(f.sensor_hole_fraction) : "—"}</b></div>
        <div><span>Depth edges</span><b>${f.depth_edge_fraction != null ? pct(f.depth_edge_fraction) : "—"}</b></div>
      </div>
      <div class="card-k">Label scores</div>
      <div class="scores">${scores}</div>
      <p class="dos-remedy">${T.remedy || ""}</p>`;
    requestAnimationFrame(() => requestAnimationFrame(() => $$("#dossier-body .bar i").forEach((i) => { i.style.width = i.dataset.w + "%"; })));
  }

  let sceneToken = 0;
  async function selectScene(idx) {
    const token = ++sceneToken;
    const sc = state.scenes[idx];
    const P = await prepScene(sc);
    if (token !== sceneToken) return;
    state.sc = sc;
    state.P = P;
    state.filter = new Set([...LABEL_ORDER, "ambiguous"]);
    $$(".scene-tab").forEach((b, i) => { b.classList.toggle("on", i === idx); b.setAttribute("aria-selected", i === idx); });
    renderSceneMetrics();
    renderFilter();
    renderRail();
    updateThreshold();
    if (state.lift) state.lift.setScene(P);
    if (state.hero) state.hero.setScene(P);
    $("#hero-scene").textContent = `${sc.sample_id} · ${sc.title.toLowerCase()}`;
    selectRegion(defaultRegion(sc));
    renderInside();
  }

  // ---------- hero ----------
  function initHero() {
    const hero = new PointCloud($("#hero-canvas"), { fov: 52, step: innerWidth < 760 ? 3 : 2, offsetX: 0.2, sizeMul: 1.25 });
    state.hero = hero;
    hero.uniforms.uBoxOn.value = 0;
    const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
    window.addEventListener("pointermove", (e) => { mouse.tx = (e.clientX / innerWidth) * 2 - 1; mouse.ty = (e.clientY / innerHeight) * 2 - 1; });
    const stateEl = $("#hero-state"), meter = $("#hero-meter");
    let lastPred = null;
    hero.onFrame = (t) => {
      // 10 s cycle: hold truth, morph, hold prediction, morph back
      const c = t % 10;
      let m;
      if (c < 3) m = 0; else if (c < 5) m = smooth((c - 3) / 2); else if (c < 8) m = 1; else m = 1 - smooth((c - 8) / 2);
      hero.uniforms.uMix.value = m;
      hero.uniforms.uGlow.value = m; // failing points light up as the prediction takes over
      const [z0, z1] = hero.zRange || [1, 4];
      hero.uniforms.uScan.value = z0 + ((t * 0.9) % 1.4) / 1.4 * (z1 - z0 + 1) - 0.5;
      mouse.x += (mouse.tx - mouse.x) * 0.04;
      mouse.y += (mouse.ty - mouse.y) * 0.04;
      const az = Math.sin(t * 0.15) * 0.32 + mouse.x * 0.28;
      const el2 = 0.08 + mouse.y * 0.12;
      const T = hero.target, R = -T.z * 1.7;
      hero.camera.position.set(T.x + Math.sin(az) * R, T.y + Math.sin(el2) * R, T.z + Math.cos(az) * R);
      hero.camera.lookAt(T);
      const pred = m > 0.5;
      if (pred !== lastPred) {
        lastPred = pred;
        stateEl.textContent = pred ? "Model prediction" : "Sensor truth";
        stateEl.classList.toggle("pred", pred);
      }
      meter.style.width = pct(m, 1);
    };
  }

  // ---------- 3D lift ----------
  function initLift() {
    const lift = new PointCloud($("#lift-canvas"), { fov: 50, step: innerWidth < 760 ? 3 : 2, sizeMul: 1.2,
      onScene: (pc) => {
        // start beside and slightly above the Kinect, so the depth difference reads at once
        const T = pc.target, d = -T.z * 2;
        pc.camera.position.set(T.x + d * 0.45, T.y + d * 0.25, T.z + d * 0.86);
        pc.controls.target.copy(T);
        pc.controls.update();
      } });
    state.lift = lift;
    lift.controls = new THREE.OrbitControls(lift.camera, lift.canvas);
    lift.controls.enableDamping = true;
    lift.controls.dampingFactor = 0.08;
    lift.controls.rotateSpeed = 0.6;
    lift.controls.minDistance = 0.3;
    lift.controls.maxDistance = 12;

    const morph = $("#morph"), play = $("#morph-play");
    let playing = !reduceMotion, dir = 1, last = 0;
    play.textContent = playing ? "❚❚" : "▶";
    morph.addEventListener("input", () => { lift.uniforms.uMix.value = +morph.value; playing = false; play.textContent = "▶"; });
    play.onclick = () => { playing = !playing; play.textContent = playing ? "❚❚" : "▶"; };
    $$("#lift .seg button").forEach((b) => b.onclick = () => {
      lift.uniforms.uColorMode.value = +b.dataset.color;
      $$("#lift .seg button").forEach((x) => x.classList.toggle("on", x === b));
    });
    $("#fail-glow").onchange = (e) => { lift.uniforms.uGlow.value = e.target.checked ? 1 : 0; };
    $("#scan").onchange = (e) => { lift.uniforms.uScanOn.value = e.target.checked ? 1 : 0; };
    lift.onFrame = (t) => {
      const dt = Math.min(0.05, t - last); last = t;
      if (playing) {
        let m = lift.uniforms.uMix.value + dir * dt * 0.28;
        if (m >= 1.15) { dir = -1; } if (m <= -0.15) { dir = 1; }
        lift.uniforms.uMix.value = m;
        morph.value = clamp(m);
      }
      const [z0, z1] = lift.zRange || [1, 4];
      lift.uniforms.uScan.value = z0 + ((t * 0.7) % 1.6) / 1.6 * (z1 - z0 + 1) - 0.5;
      lift.controls.update();
    };
  }

  // ---------- labels ----------
  function renderLabels() {
    const host = $("#label-grid");
    LABEL_RESULTS.forEach((L) => {
      const T = state.tax[L.id] || {};
      const c = el("div", "card lab reveal");
      c.style.setProperty("--c", LABELS[L.id].color);
      const rows = L.au
        ? `<div class="auroc">${L.au.map((v, i) => `<div class="au-row${v == null ? " na" : ""}"><span class="k">${AU_SETS[i]}</span><div class="au-track"><i style="--w:${v == null ? 0 : v * 100}%"></i><span class="chance"></span></div><span class="v">${v == null ? "—" : v.toFixed(2)}</span></div>`).join("")}</div>`
        : "";
      c.innerHTML = `<div class="lab-top"><span class="lab-id">${L.id}</span><span class="lab-verdict ${L.verdict[1]}">${L.verdict[0]}</span></div>
        <h3>${LABELS[L.id].name}</h3>
        <p>${T.desc || ""}</p>
        ${rows}
        <div class="au-note">${L.au ? "AUROC · line marks chance (0.5) · " : ""}${L.note}</div>
        <div class="lab-meta">Cues: ${T.metric_name || "—"}<br>${T.formula || ""}</div>`;
      if (finePointer) {
        c.addEventListener("pointermove", (e) => {
          const r = c.getBoundingClientRect();
          c.style.setProperty("--mx", e.clientX - r.left + "px");
          c.style.setProperty("--my", e.clientY - r.top + "px");
        });
      }
      host.appendChild(c);
    });
  }

  // ---------- forest plot ----------
  function renderForest() {
    const lo = -0.025, hi = 0.055, w = 1000, left = 400, right = 190, rowH = 64, top = 24;
    const sg = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(3);
    const h = top + INTERVENTIONS.length * rowH + 50;
    const sx = (v) => left + ((v - lo) / (hi - lo)) * (w - left - right);
    let s = `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Forest plot of four image-edit tests">`;
    s += `<line class="axis" x1="${left}" x2="${w - right}" y1="${h - 34}" y2="${h - 34}"/>`;
    for (let v = -0.02; v <= 0.0501; v += 0.01) {
      const x = sx(v);
      const r = Math.round(v * 100) / 100;
      s += `<line class="axis" x1="${x}" x2="${x}" y1="${h - 34}" y2="${h - 29}"/><text class="tick" x="${x}" y="${h - 12}" text-anchor="middle">${r > 0 ? "+" : r < 0 ? "−" : ""}${Math.abs(r).toFixed(2)}</text>`;
    }
    s += `<line class="zero" x1="${sx(0)}" x2="${sx(0)}" y1="${top - 6}" y2="${h - 34}"/>`;
    s += `<text class="tick" x="${sx(0) - 8}" y="${top - 8}" text-anchor="end">← helped labelled regions more</text>`;
    INTERVENTIONS.forEach((t, i) => {
      const y = top + i * rowH + rowH / 2;
      const col = t.ok ? "var(--good)" : "var(--muted)";
      s += `<text class="row-label" x="0" y="${y - 3}">${t.edit} → <tspan fill="${LABELS[t.label].color}">${LABELS[t.label].name}</tspan></text>`;
      s += `<text class="row-sub" x="0" y="${y + 16}">${sg(t.est)} [${sg(t.lo)}, ${sg(t.hi)}] · Holm p = ${t.p}</text>`;
      s += `<line class="ci grow" stroke="${col}" x1="${sx(t.lo)}" x2="${sx(t.hi)}" y1="${y}" y2="${y}" style="transition-delay:${i * 0.15}s"/>`;
      s += `<circle class="pt pop" fill="${col}" cx="${sx(t.est)}" cy="${y}" r="8" style="transition-delay:${0.8 + i * 0.15}s"/>`;
      s += `<text class="res pop" x="${w}" y="${y + 4}" text-anchor="end" fill="${col}" style="transition-delay:${0.9 + i * 0.15}s">${t.tag}</text>`;
    });
    s += "</svg>";
    $("#forest").innerHTML = s;
  }

  // ---------- leaderboard ----------
  let metricIdx = 0;
  function renderBoard() {
    const seg = $("#metric-seg");
    METRICS.forEach((m, i) => {
      const b = el("button", i === 0 ? "on" : "", m.name);
      b.onclick = () => { metricIdx = i; $$("#metric-seg button").forEach((x, j) => x.classList.toggle("on", j === i)); layoutBoard(); };
      seg.appendChild(b);
    });
    const board = $("#board");
    MODELS.forEach((m) => {
      const r = el("div", "brow" + (m.me ? " me" : ""));
      r.innerHTML = `<span class="rank"></span><span class="name">${m.name}</span><div class="btrack"><i></i></div><span class="val"></span>`;
      m._el = r;
      board.appendChild(r);
    });
    layoutBoard(true);
  }
  function layoutBoard(initial) {
    const M = METRICS[metricIdx];
    const rowH = innerWidth < 760 ? 58 : 46;
    const sorted = MODELS.slice().sort((a, b) => (M.lower ? a.v[metricIdx] - b.v[metricIdx] : b.v[metricIdx] - a.v[metricIdx]));
    const max = Math.max(...MODELS.map((m) => m.v[metricIdx]));
    $("#board").style.height = MODELS.length * rowH + 36 + "px";
    sorted.forEach((m, rank) => {
      const v = m.v[metricIdx];
      // lower-is-better metrics: bar is the error itself; δ₁: bar shows distance above 0.88 so differences read
      const w = M.lower ? (v / max) * 100 : ((v - 0.88) / (1 - 0.88)) * 100;
      m._el.style.transform = `translateY(${18 + rank * rowH}px)`;
      $(".rank", m._el).textContent = rank + 1;
      $(".val", m._el).textContent = M.fmt(v);
      $(".btrack i", m._el).style.width = initial ? "0%" : clamp(w, 2, 100) + "%";
      m._w = clamp(w, 2, 100);
    });
  }

  // ---------- null results ----------
  function renderHonest() {
    const fp = [["NYU", -0.29], ["DIODE", -0.12], ["iBims-1", -0.25]];
    $("#fp").innerHTML = fp.map(([k, v]) => `<div class="fp-row"><span class="k">${k}</span><div class="fp-track"><i style="--w:${(Math.abs(v) / 0.35) * 100}%"></i><span class="zero"></span></div><span class="v">${v.toFixed(2)}</span></div>`).join("");
    const ft = [
      { k: "Boundaries", g: 0.113, r: 0.093, d: "+0.020 [+0.018, +0.022]" },
      { k: "Walls", g: 0.048, r: 0.044, d: "+0.003 [+0.002, +0.005]" },
    ];
    $("#ft").innerHTML = ft.map((f) => `<div class="ft-row"><span class="k">${f.k}</span><div class="ft-pair">
        <div class="ft-bar g"><i style="--w:${(f.g / 0.113) * 62}%"></i><span>guided ${f.g.toFixed(3)}</span></div>
        <div class="ft-bar r"><i style="--w:${(f.r / 0.113) * 62}%"></i><span>random ${f.r.toFixed(3)}</span></div>
      </div><span class="v" style="color:var(--hot)">${f.d.split(" ")[0]}</span></div>`).join("") +
      `<p class="mono muted small" style="margin:6px 0 0">Guided minus random: ${ft.map((f) => `${f.k.toLowerCase()} ${f.d}`).join(" · ")}</p>`;
  }

  // ---------- inside the model ----------
  const FM_KEYS = ["layer_3", "layer_6", "layer_9", "layer_12"];
  let fmImgs = null;
  function initInside() {
    const stage = $(".fm-stage");
    const a = $("#fm-map");
    const b = a.cloneNode();
    b.id = "fm-map-b";
    b.style.mixBlendMode = "screen";
    stage.insertBefore(b, $("#fm-layer"));
    fmImgs = [a, b];
    $("#fm-ticks").innerHTML = FM_KEYS.map((k, i) => `<span data-i="${i}">L${k.split("_")[1]}</span>`).join("");
    $$("#fm-ticks span").forEach((s) => s.onclick = () => animateScrub(+s.dataset.i));
    $("#fm-scrub").addEventListener("input", updateInside);
    $("#fm-blend").addEventListener("input", updateInside);
  }
  function animateScrub(to) {
    const s = $("#fm-scrub"), from = +s.value, t0 = performance.now();
    const step = (t) => {
      const k = clamp((t - t0) / 600);
      s.value = from + (to - from) * ease(k);
      updateInside();
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  function renderInside() {
    if (!fmImgs || !state.sc) return;
    $("#fm-photo").src = state.sc.images.rgb;
    updateInside(true);
  }
  function updateInside(force) {
    const sc = state.sc;
    if (!sc) return;
    const v = +$("#fm-scrub").value, blend = +$("#fm-blend").value;
    const i0 = Math.min(2, Math.floor(v)), f = v - i0;
    const k0 = FM_KEYS[i0], k1 = FM_KEYS[i0 + 1];
    const [a, b] = fmImgs;
    if (force === true || a.dataset.k !== sc.sample_id + k0) { a.src = sc.feature_maps[k0].image; a.dataset.k = sc.sample_id + k0; }
    if (force === true || b.dataset.k !== sc.sample_id + k1) { b.src = sc.feature_maps[k1].image; b.dataset.k = sc.sample_id + k1; }
    a.style.opacity = blend * (1 - f);
    b.style.opacity = blend * f;
    const near = Math.round(v), fm = sc.feature_maps[FM_KEYS[near]];
    $("#fm-layer").textContent = `Block ${FM_KEYS[near].split("_")[1]} of 12`;
    $("#fm-desc").textContent = fm.description;
    $$("#fm-ticks span").forEach((s, i) => s.classList.toggle("on", i === near));
    const [th, tw, ch] = fm.tokens_shape;
    $("#fm-meta").textContent = `${th} × ${tw} patch tokens · ${ch} channels · activation ${fm.min_act.toFixed(2)} to ${fm.max_act.toFixed(2)}`;
  }

  // ---------- page chrome ----------
  function initChrome() {
    const nav = $("#nav"), fill = $("#scroll-fill");
    const links = $$(".nav-links a[href^='#']");
    const sections = links.map((a) => $(a.getAttribute("href")));
    const onScroll = () => {
      const y = scrollY;
      nav.classList.toggle("scrolled", y > 40);
      fill.style.width = pct(y / Math.max(1, document.documentElement.scrollHeight - innerHeight));
      let cur = -1;
      sections.forEach((s, i) => { if (s && s.getBoundingClientRect().top < innerHeight * 0.4) cur = i; });
      links.forEach((a, i) => a.classList.toggle("active", i === cur));
      litStatement();
    };
    addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    if (finePointer && !reduceMotion) {
      $$(".magnetic").forEach((b) => {
        b.addEventListener("pointermove", (e) => {
          const r = b.getBoundingClientRect();
          b.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * 0.22}px, ${(e.clientY - r.top - r.height / 2) * 0.3}px)`;
        });
        b.addEventListener("pointerleave", () => { b.style.transform = ""; });
      });
    }

    $("#copy-btn").onclick = async () => {
      const text = $("#term-code").textContent.split("\n").filter((l) => !l.startsWith("#")).join("\n").trim();
      try { await navigator.clipboard.writeText(text); toast("Commands copied"); }
      catch (e) { toast("Copy failed: select the text instead"); }
    };
  }

  function splitStatement() {
    const s = $("#statement");
    const hl = new Set(["97.3%", "regions,", "name,", "measure", "test."]);
    s.innerHTML = s.textContent.trim().split(/\s+/).map((w) => `<span class="w${hl.has(w) ? " hl" : ""}">${w}</span>`).join(" ");
  }
  function litStatement() {
    const s = $("#statement");
    if (!s) return;
    const r = s.getBoundingClientRect();
    const f = clamp((innerHeight * 0.85 - r.top) / (r.height + innerHeight * 0.35));
    const words = s.children;
    const n = Math.round(f * words.length);
    for (let i = 0; i < words.length; i++) words[i].classList.toggle("lit", reduceMotion || i < n);
  }

  function observeReveal() {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        e.target.classList.add("in");
        io.unobserve(e.target);
        if (e.target.classList.contains("stat")) countUp($(".count", e.target));
        if (e.target.id === "board") requestAnimationFrame(() => MODELS.forEach((m) => { $(".btrack i", m._el).style.width = m._w + "%"; }));
      });
    }, { threshold: 0.15, rootMargin: "0px 0px -6% 0px" });
    $$(".reveal").forEach((e, i) => {
      if (!e.style.transitionDelay && e.parentElement) {
        const sib = Array.from(e.parentElement.children).filter((x) => x.classList.contains("reveal"));
        e.style.transitionDelay = Math.min(sib.indexOf(e), 6) * 0.07 + "s";
      }
      io.observe(e);
    });
  }

  function countUp(e) {
    if (!e) return;
    const to = +e.dataset.to, t0 = performance.now(), dur = 1400;
    const step = (t) => {
      const k = clamp((t - t0) / dur);
      e.textContent = Math.round(to * ease(k)).toLocaleString("en-US");
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function wipeIntro() {
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      const t0 = performance.now();
      const step = (t) => {
        const k = clamp((t - t0) / 1400);
        state.wipeX = W * (0.92 - 0.42 * ease(k));
        requestDraw();
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }, { threshold: 0.4 });
    io.observe($("#viewer"));
  }

  // ---------- boot ----------
  async function boot() {
    document.body.classList.add("loading");
    splitStatement();
    initChrome();
    renderForest();

    let data;
    try { data = await loadData(); }
    catch (e) {
      $("#loader-text").textContent = "Could not load scene data. Run python web_app/fetch_assets.py";
      return;
    }
    state.data = data;
    state.scenes = data.scenes;
    data.taxonomy.forEach((t) => { state.tax[t.id] = t; });

    renderLabels();
    renderBoard();
    renderHonest();
    initExplorer();
    initInside();

    if (window.VEO.webglOK()) {
      try { initHero(); initLift(); }
      catch (e) { console.warn("WebGL unavailable", e); state.hero = state.lift = null; }
    }
    if (!state.hero) {
      $("#hero").style.background = `center/cover no-repeat url(${data.scenes[0].images.rgb})`;
      $("#lift").style.display = "none";
      $(".hero-hud").style.display = "none";
    }

    await selectScene(0);
    observeReveal();
    wipeIntro();

    $("#loader-fill").style.width = "100%";
    $("#loader-pct").textContent = "100%";
    setTimeout(() => {
      $("#loader").classList.add("done");
      document.body.classList.remove("loading");
      $$(".hero .reveal").forEach((e, i) => setTimeout(() => e.classList.add("in"), 150 + i * 120));
    }, 250);
  }

  boot();
})();
