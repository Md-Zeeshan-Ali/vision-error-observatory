/* Vision Error Observatory: Model Lab.
   A scroll-driven walk through Depth Anything V2 Small. The photo, activation maps, prediction and
   sensor depth are this scene's exported data; the motion between stages is illustration. */
(() => {
  "use strict";

  const { W, H, CAM, clamp, smooth, mkCanvas, TURBO, INFERNO } = window.VEO;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  const COLS = 49, ROWS = 37, N = COLS * ROWS; // 518 × 686 input in 14-pixel patches
  const SW = 6.4, SH = (SW * ROWS) / COLS; // flat patch grid in world units
  const S = 1.5; // world units per metre once the patches stand in 3D
  const FM_KEYS = ["layer_3", "layer_6", "layer_9", "layer_12"];
  const LAST = 8; // index of the final step
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // camera [position, look-at] reached at the middle of each step
  const SHOTS = [
    [[0, 0, 9.0], [0, 0, 0]],
    [[0, 0, 8.0], [0, 0, 0]],
    [[2.8, 1.6, 8.2], [0, 0, 0]],
    [[0, 0.6, 13.5], [0, 0, 0]],
    [[6.5, 3.0, 6.5], [0, 0, -1.5]],
    [[0, 0.4, 10.0], [0, 0, 0.6]],
    [[0, 0, 8.6], [0, 0, 0]],
    [[3.2, 1.8, 7.6], [0, 0, 0]],
    [[-3.2, 1.7, 7.4], [0, 0, 0]],
  ];

  const VERT = `
    uniform float uSplit, uSeq, uRelief, uGT, uTime, uZMid;
    uniform vec4 uCam;
    attribute vec2 aCell;
    attribute float aRnd, aDP, aDG;
    varying vec2 vUv, vLocal;
    void main() {
      float col = aCell.x, row = aCell.y;
      vec2 g = vec2((col + 0.5) / ${COLS}.0 - 0.5, 0.5 - (row + 0.5) / ${ROWS}.0) * vec2(${SW.toFixed(4)}, ${SH.toFixed(4)});
      vec3 pos = vec3(g * (1.0 + 0.16 * uSplit), 0.0);
      float size = ${(SW / COLS).toFixed(5)} * (1.0 - 0.22 * uSplit);

      // token sequence: a slowly turning helix
      float i = row * ${COLS}.0 + col;
      float a = i * 0.055 + uTime * 0.15;
      vec3 helix = vec3(cos(a) * 2.6, (i / ${N}.0 - 0.5) * 7.0, sin(a) * 2.6);
      float ms = clamp(uSeq * 1.6 - aRnd * 0.6, 0.0, 1.0);
      ms = ms * ms * (3.0 - 2.0 * ms);
      pos = mix(pos, helix, ms);
      size = mix(size, 0.09, ms);

      // relief: stand each patch at its depth, unprojected with the Kinect intrinsics
      float d = mix(aDP, aDG, uGT);
      vec2 px = vec2((col + 0.5) / ${COLS}.0 * ${W}.0, (row + 0.5) / ${ROWS}.0 * ${H}.0);
      vec3 world = vec3((px.x - uCam.z) * d / uCam.x, -(px.y - uCam.w) * d / uCam.y, -d + uZMid) * ${S.toFixed(2)};
      float mr = clamp(uRelief * 1.5 - aRnd * 0.5, 0.0, 1.0);
      mr = mr * mr * (3.0 - 2.0 * mr);
      pos = mix(pos, world, mr);
      size = mix(size, ${(W / COLS).toFixed(4)} * d / uCam.x * ${S.toFixed(2)} * 1.04, mr);

      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos + vec3(position.xy * size, 0.0), 1.0);
      vUv = vec2((col + uv.x) / ${COLS}.0, 1.0 - (row + 1.0 - uv.y) / ${ROWS}.0);
      vLocal = uv;
    }`;
  const FRAG = `
    uniform sampler2D uPhoto, uFeatA, uFeatB, uDepth, uErr;
    uniform float uFeatMix, uFeatF, uDepthMix, uErrMix, uGrid, uSplit;
    varying vec2 vUv, vLocal;
    void main() {
      vec3 c = texture2D(uPhoto, vUv).rgb;
      vec3 f = mix(texture2D(uFeatA, vUv).rgb, texture2D(uFeatB, vUv).rgb, uFeatF);
      c = mix(c, f, uFeatMix);
      c = mix(c, texture2D(uDepth, vUv).rgb, uDepthMix);
      c = mix(c, texture2D(uErr, vUv).rgb, uErrMix);
      vec2 e = min(vLocal, 1.0 - vLocal);
      float edge = 1.0 - smoothstep(0.0, 0.07, min(e.x, e.y));
      c = mix(c, vec3(1.0), edge * max(uGrid, uSplit) * 0.32);
      gl_FragColor = vec4(c, 1.0);
    }`;

  const state = { scenes: [], sc: null, p: 0, pShow: 0, orbit: false };
  const T = {}; // three.js objects

  // ---------- textures from scene data ----------
  const texLoader = () => new THREE.TextureLoader();
  function canvasTex(cv) { const t = new THREE.CanvasTexture(cv); t.minFilter = THREE.LinearFilter; return t; }

  function depthTextures(P) {
    // prediction as relative inverse depth (what the head outputs), and the error against the sensor
    const dc = mkCanvas(W, H), ec = mkCanvas(W, H);
    const dx = dc.getContext("2d"), ex = ec.getContext("2d");
    const dd = dx.createImageData(W, H), ed = ex.createImageData(W, H);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < W * H; i++) { const v = 1 / Math.max(P.pd[i], 0.05); if (v < lo) lo = v; if (v > hi) hi = v; }
    for (let i = 0; i < W * H; i++) {
      const o = i * 4;
      let k = Math.round(clamp((1 / Math.max(P.pd[i], 0.05) - lo) / (hi - lo || 1)) * 255) * 3;
      dd.data[o] = TURBO[k]; dd.data[o + 1] = TURBO[k + 1]; dd.data[o + 2] = TURBO[k + 2]; dd.data[o + 3] = 255;
      if (P.crop[i]) {
        k = Math.round(clamp(P.lr[i] / Math.LN2) * 255) * 3;
        ed.data[o] = INFERNO[k]; ed.data[o + 1] = INFERNO[k + 1]; ed.data[o + 2] = INFERNO[k + 2];
      } else {
        ed.data[o] = ed.data[o + 1] = ed.data[o + 2] = 12;
      }
      ed.data[o + 3] = 255;
    }
    dx.putImageData(dd, 0, 0);
    ex.putImageData(ed, 0, 0);
    return { depth: canvasTex(dc), err: canvasTex(ec) };
  }

  // mean truth and prediction (metres) under each patch
  function patchDepths(P) {
    const dp = new Float32Array(N), dg = new Float32Array(N);
    let sum = 0;
    for (let r = 0; r < ROWS; r++) {
      const y0 = Math.floor((r * H) / ROWS), y1 = Math.floor(((r + 1) * H) / ROWS);
      for (let c = 0; c < COLS; c++) {
        const x0 = Math.floor((c * W) / COLS), x1 = Math.floor(((c + 1) * W) / COLS);
        let sp = 0, sg = 0, n = 0;
        for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const i = y * W + x; sp += P.pd[i]; sg += P.gt[i]; n++; }
        const k = r * COLS + c;
        dp[k] = sp / n; dg[k] = sg / n;
        sum += dp[k];
      }
    }
    return { dp, dg, mid: sum / N };
  }

  // ---------- three.js scene ----------
  function initThree() {
    const canvas = $("#lab-canvas");
    T.pr = Math.min(window.devicePixelRatio || 1, 2);
    T.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    T.renderer.setPixelRatio(T.pr);
    T.renderer.setClearColor(0x06070b, 1);
    T.scene = new THREE.Scene();
    T.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 80);
    T.camera.position.set(0, 0, 7.2);
    T.look = new THREE.Vector3();

    const blank = canvasTex(mkCanvas(2, 2));
    T.uniforms = {
      uSplit: { value: 0 }, uSeq: { value: 0 }, uRelief: { value: 0 }, uGT: { value: 0 }, uTime: { value: 0 }, uZMid: { value: 2.5 },
      uCam: { value: new THREE.Vector4(CAM.fx, CAM.fy, CAM.cx, CAM.cy) },
      uPhoto: { value: blank }, uFeatA: { value: blank }, uFeatB: { value: blank }, uDepth: { value: blank }, uErr: { value: blank },
      uFeatMix: { value: 0 }, uFeatF: { value: 0 }, uDepthMix: { value: 0 }, uErrMix: { value: 0 }, uGrid: { value: 0 },
    };

    const base = new THREE.PlaneGeometry(1, 1);
    const geo = new THREE.InstancedBufferGeometry();
    geo.index = base.index;
    geo.setAttribute("position", base.getAttribute("position"));
    geo.setAttribute("uv", base.getAttribute("uv"));
    geo.instanceCount = N;
    const cell = new Float32Array(N * 2), rnd = new Float32Array(N);
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      const k = r * COLS + c;
      cell[k * 2] = c; cell[k * 2 + 1] = r;
      rnd[k] = (c / COLS) * 0.7 + Math.random() * 0.3;
    }
    geo.setAttribute("aCell", new THREE.InstancedBufferAttribute(cell, 2));
    geo.setAttribute("aRnd", new THREE.InstancedBufferAttribute(rnd, 1));
    T.aDP = new THREE.InstancedBufferAttribute(new Float32Array(N), 1);
    T.aDG = new THREE.InstancedBufferAttribute(new Float32Array(N), 1);
    geo.setAttribute("aDP", T.aDP);
    geo.setAttribute("aDG", T.aDG);
    T.tiles = new THREE.Mesh(geo, new THREE.ShaderMaterial({ uniforms: T.uniforms, vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide }));
    T.tiles.frustumCulled = false;
    T.scene.add(T.tiles);

    // class token
    T.cls = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.16, 0.16), new THREE.MeshBasicMaterial({ color: 0xffb547, transparent: true }));
    T.scene.add(T.cls);

    // twelve transformer blocks as frames the patch grid passes through
    T.frames = [];
    for (let k = 0; k < 12; k++) {
      const w = SW * 1.12, h = SH * 1.12;
      const pts = [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]].map(([x, y]) => new THREE.Vector3(x, y, 0));
      const frame = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: [2, 5, 8, 11].includes(k) ? 0xff5a36 : 0x59d0ff, transparent: true, opacity: 0 }));
      const label = makeLabel(`BLOCK ${k + 1}`, [2, 5, 8, 11].includes(k) ? "#ff5a36" : "#59d0ff");
      label.position.set(-w / 2 + 0.55, h / 2 + 0.18, 0);
      frame.add(label);
      T.scene.add(frame);
      T.frames.push(frame);
    }

    // four decoder taps
    T.taps = FM_KEYS.map((k, i) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 0.75), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }));
      const label = makeLabel(`BLOCK ${[3, 6, 9, 12][i]}`, "#eeece6");
      label.position.set(0, 0.52, 0);
      label.scale.multiplyScalar(0.8);
      m.add(label);
      T.scene.add(m);
      return m;
    });

    T.controls = new THREE.OrbitControls(T.camera, canvas);
    T.controls.enabled = false;
    T.controls.enableZoom = false; // keep the wheel for scrolling the page
    T.controls.enableDamping = true;
    T.controls.rotateSpeed = 0.6;

    new ResizeObserver(resize).observe(document.body);
    resize();
    requestAnimationFrame(frame);
  }

  function makeLabel(text, color) {
    const c = mkCanvas(256, 48), x = c.getContext("2d");
    x.font = "600 26px 'JetBrains Mono', monospace";
    x.fillStyle = color;
    x.fillText(text, 4, 34);
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: canvasTex(c), transparent: true, depthWrite: false }));
    s.scale.set(1.1, 0.21, 1);
    return s;
  }

  function resize() {
    const w = innerWidth, h = innerHeight;
    T.renderer.setSize(w, h, false);
    T.camera.aspect = w / h;
    if (w > 900) T.camera.setViewOffset(w, h, -w * 0.17, 0, w, h);
    else T.camera.setViewOffset(w, h, 0, h * 0.2, w, h);
    T.camera.updateProjectionMatrix();
  }

  const r = (p, a, b) => smooth(clamp((p - a) / (b - a)));
  const lerp3 = (a, b, f) => a.map((v, i) => v + (b[i] - v) * f);

  function frame(ms) {
    requestAnimationFrame(frame);
    const t = ms / 1000;
    state.pShow += (state.p - state.pShow) * (reduceMotion ? 1 : 0.08);
    const p = state.pShow;
    const U = T.uniforms;
    U.uTime.value = t;
    U.uGrid.value = r(p, 0.9, 1.45);
    U.uSplit.value = r(p, 1.9, 2.45) * (1 - r(p, 5.9, 6.45));
    U.uSeq.value = r(p, 2.9, 3.45) * (1 - r(p, 3.9, 4.3));
    U.uFeatMix.value = r(p, 3.95, 4.25) * (1 - r(p, 5.9, 6.45));
    U.uDepthMix.value = r(p, 5.9, 6.45);
    U.uRelief.value = r(p, 6.9, 7.45);
    U.uGT.value = r(p, 7.9, 8.4);
    U.uErrMix.value = r(p, 7.9, 8.45);

    // feature maps cross-fade as blocks 3, 6, 9 and 12 pass
    const b = clamp((p - 3.95) / 0.8);
    const fi = clamp(((b * 13 - 3) / 9) * 3, 0, 3);
    const i0 = Math.min(2, Math.floor(fi));
    if (T.fm && U.uFeatA.value !== T.fm[i0]) { U.uFeatA.value = T.fm[i0]; U.uFeatB.value = T.fm[i0 + 1]; }
    U.uFeatF.value = fi - i0;

    const seq = U.uSeq.value;
    const a0 = -6 * 0.055 + t * 0.15;
    T.cls.position.set(Math.cos(a0) * 2.6, -3.5 - 0.1, Math.sin(a0) * 2.6);
    T.cls.rotation.set(t, t * 0.7, 0);
    T.cls.material.opacity = clamp(seq * 2 - 1);
    T.cls.visible = seq > 0.02;

    const vis = r(p, 3.85, 4.05) * (1 - r(p, 4.8, 5.0));
    T.frames.forEach((f, k) => {
      const z = -(k + 1) * 0.9 + b * 12.5 * 0.9;
      f.position.z = z;
      const o = vis * clamp(1 - Math.abs(z) / 5) * (z > 0 ? clamp(1 - z / 2.5) : 1);
      f.material.opacity = o;
      f.children[0].material.opacity = o;
      f.visible = o > 0.01;
    });

    const tv = r(p, 4.85, 5.3) * (1 - r(p, 5.9, 6.35));
    const merge = r(p, 5.6, 6.2);
    const sizes = [1.6, 1.45, 1.3, 1.15]; // DPT resamples block 3 largest and block 12 smallest
    T.taps.forEach((m, i) => {
      const x = ((i - 1.5) * 1.75 + 0.35) * (1 - merge);
      m.position.set(x, 0.1 + Math.sin(t * 0.8 + i) * 0.05 * (1 - merge), 2.0 * (1 - merge));
      m.scale.setScalar(sizes[i] * (1 - 0.4 * merge));
      m.material.opacity = tv;
      m.children[0].material.opacity = tv;
      m.visible = tv > 0.01;
    });
    // the grid steps back while the taps are in front
    T.tiles.position.z = -4 * tv;
    T.tiles.scale.setScalar(1 - 0.55 * tv);

    // camera: follow the shots unless the visitor is orbiting the final room
    const wantOrbit = p > LAST - 0.35;
    if (wantOrbit !== state.orbit) {
      state.orbit = wantOrbit;
      T.controls.enabled = wantOrbit;
      document.body.classList.toggle("orbit", wantOrbit);
      if (wantOrbit) T.controls.target.copy(T.look);
    }
    if (state.orbit) {
      T.controls.update();
    } else {
      const q = clamp(p - 0.5, 0, LAST);
      const k0 = Math.floor(q), k1 = Math.min(LAST, k0 + 1), f = smooth(q - k0);
      const look = lerp3(SHOTS[k0][1], SHOTS[k1][1], f);
      // narrow screens: back away so the 6.4-unit grid still fits across
      const far = Math.max(1, 1.1 / T.camera.aspect);
      const pos = lerp3(SHOTS[k0][0], SHOTS[k1][0], f).map((v, i) => look[i] + (v - look[i]) * far);
      const drift = reduceMotion ? 0 : 0.12;
      T.camera.position.lerp(new THREE.Vector3(pos[0] + Math.sin(t * 0.3) * drift, pos[1] + Math.cos(t * 0.25) * drift * 0.6, pos[2]), 0.1);
      T.look.lerp(new THREE.Vector3(...look), 0.1);
      T.camera.lookAt(T.look);
    }
    T.renderer.render(T.scene, T.camera);
  }

  // ---------- scenes ----------
  async function selectScene(idx) {
    const sc = state.scenes[idx];
    state.sc = sc;
    $$("#lab-scenes button").forEach((b, i) => { b.classList.toggle("on", i === idx); b.setAttribute("aria-selected", i === idx); });
    renderFacts(sc);
    if (!T.renderer) return;
    const P = await window.VEO.prepScene(sc);
    if (state.sc !== sc) return;
    const L = texLoader();
    const old = [T.uniforms.uPhoto.value, ...(T.fm || []), T.uniforms.uDepth.value, T.uniforms.uErr.value];
    T.uniforms.uPhoto.value = L.load(sc.images.rgb);
    T.fm = FM_KEYS.map((k) => L.load(sc.feature_maps[k].image));
    T.uniforms.uFeatA.value = T.fm[0];
    T.uniforms.uFeatB.value = T.fm[1];
    T.taps.forEach((m, i) => { m.material.map = T.fm[i]; m.material.needsUpdate = true; });
    const { depth, err } = depthTextures(P);
    T.uniforms.uDepth.value = depth;
    T.uniforms.uErr.value = err;
    old.forEach((t) => t && t.dispose && t.dispose());

    const { dp, dg, mid } = patchDepths(P);
    T.aDP.array.set(dp); T.aDP.needsUpdate = true;
    T.aDG.array.set(dg); T.aDG.needsUpdate = true;
    T.uniforms.uZMid.value = mid;
  }

  function renderFacts(sc) {
    const a = sc.alignment, m = sc.metrics;
    $("#align-facts").innerHTML = `<span><b>s = ${a.scale.toFixed(4)}</b></span><span><b>t = ${a.shift.toFixed(4)}</b></span><span>this scene: <b>${sc.sample_id}</b></span>`;
    $("#audit-facts").innerHTML = `<span>AbsRel <b>${m.abs_rel.toFixed(3)}</b></span><span>δ₁ <b>${(m.delta1 * 100).toFixed(1)}%</b></span><span>failing <b>${(m.failure_pixel_fraction * 100).toFixed(1)}%</b></span><span><b>${m.total_regions}</b> regions</span>`;
    $("#fm-legend").innerHTML = FM_KEYS.map((k, i) => `<figure><img alt="Activation map after block ${[3, 6, 9, 12][i]}" src="${sc.feature_maps[k].image}"><figcaption>block ${[3, 6, 9, 12][i]}</figcaption></figure>`).join("");
  }

  // ---------- scroll ----------
  function initScroll() {
    const steps = $$(".step");
    const hudSteps = $("#hud-steps");
    hudSteps.innerHTML = steps.map(() => "<i></i>").join("");
    let active = -1;
    const onScroll = () => {
      const mid = innerHeight * 0.5;
      let p = 0;
      steps.forEach((s, i) => {
        const b = s.getBoundingClientRect();
        if (b.top <= mid) p = i + clamp((mid - b.top) / b.height);
      });
      state.p = Math.min(p, LAST + 0.99);
      const a = Math.min(LAST, Math.floor(state.p));
      if (a !== active) {
        active = a;
        steps.forEach((s, i) => s.classList.toggle("active", i === a));
        $("#hud-tensor").textContent = steps[a].dataset.tensor;
        $("#hud-stage").textContent = steps[a].dataset.stage;
        $$("i", hudSteps).forEach((e, i) => e.classList.toggle("on", i <= a));
      }
      $("#scroll-fill").style.width = (scrollY / Math.max(1, document.documentElement.scrollHeight - innerHeight)) * 100 + "%";
    };
    addEventListener("scroll", onScroll, { passive: true });
    addEventListener("resize", onScroll);
    onScroll();
  }

  // ---------- boot ----------
  async function boot() {
    document.body.classList.add("loading");
    let data;
    try {
      data = await window.VEO.loadData((f) => {
        $("#loader-fill").style.width = Math.round(f * 100) + "%";
        $("#loader-pct").textContent = Math.round(f * 100) + "%";
      });
    } catch (e) {
      $("#loader-text").textContent = "Could not load scene data. Run python web_app/fetch_assets.py";
      return;
    }
    state.scenes = data.scenes;
    const host = $("#lab-scenes");
    data.scenes.forEach((sc, i) => {
      const b = document.createElement("button");
      b.setAttribute("role", "tab");
      b.title = `${sc.sample_id} · ${sc.title}`;
      b.innerHTML = `<img alt="${sc.title}" src="${sc.images.rgb}">`;
      b.onclick = () => selectScene(i);
      host.appendChild(b);
    });

    if (!window.VEO.webglOK()) {
      $("#lab-canvas").style.background = `center/cover no-repeat url(${data.scenes[0].images.rgb})`;
    } else {
      initThree();
    }
    initScroll();
    await selectScene(0);

    $("#loader-fill").style.width = "100%";
    $("#loader-pct").textContent = "100%";
    setTimeout(() => { $("#loader").classList.add("done"); document.body.classList.remove("loading"); }, 250);
  }

  boot();
})();
