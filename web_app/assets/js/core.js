/* Vision Error Observatory: helpers shared by index.html and model_lab.html.
   Loads data/app_data_v2.js and decodes its depth maps; exposes everything on window.VEO. */
(() => {
  "use strict";

  const W = 640, H = 480;
  const CROP = { x0: 41, y0: 45, x1: 601, y1: 471 }; // NYU evaluation crop
  const CAM = { fx: 518.8579, fy: 519.4696, cx: 325.5824, cy: 253.7362 }; // NYU Kinect intrinsics
  const DATA_BYTES = 9277169;

  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  const smooth = (t) => t * t * (3 - 2 * t);
  const mkCanvas = (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h });
  const loadImg = (src) => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });

  function polyLUT(coef) {
    // coef[channel] = polynomial coefficients, lowest order first
    const lut = new Uint8ClampedArray(256 * 3);
    for (let k = 0; k < 256; k++) {
      const t = k / 255;
      for (let ch = 0; ch < 3; ch++) {
        const c = coef[ch];
        let v = 0;
        for (let j = c.length - 1; j >= 0; j--) v = v * t + c[j];
        lut[k * 3 + ch] = v * 255;
      }
    }
    return lut;
  }
  const TURBO = polyLUT([
    [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
    [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
    [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
  ]);
  const INFERNO_C = [
    [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
    [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
    [11.60249308247187, -3.972853965665698, -15.9423941062914],
    [-41.70399613139459, 17.43639888205313, 44.35414519872813],
    [77.162935699427, -33.40235894210092, -81.80730925738993],
    [-71.31942824499214, 32.62606426397723, 73.20951985803202],
    [25.13112622477341, -12.24266895238567, -23.07032500287172],
  ];
  const INFERNO = polyLUT([0, 1, 2].map((ch) => INFERNO_C.map((c) => c[ch])));
  const INFERNO_GLSL = `
    vec3 inferno(float t) {
      ${INFERNO_C.map((c, i) => `const vec3 c${i} = vec3(${c.join(",")});`).join("\n      ")}
      return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
    }`;
  const lutCss = (lut) => {
    const stops = [];
    for (let k = 0; k <= 8; k++) {
      const i = Math.round((k / 8) * 255) * 3;
      stops.push(`rgb(${lut[i]},${lut[i + 1]},${lut[i + 2]}) ${k * 12.5}%`);
    }
    return `linear-gradient(90deg, ${stops.join(",")})`;
  };

  function decodeDepth(b64) {
    const bin = atob(b64);
    const n = bin.length >> 1;
    const out = new Float32Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 2) out[i] = (bin.charCodeAt(j) | (bin.charCodeAt(j + 1) << 8)) / 1000;
    return out;
  }

  // Fetch the scene data with a progress callback; falls back to a script tag (file://, old browsers).
  async function loadData(onProgress = () => {}) {
    try {
      const res = await fetch("data/app_data_v2.js");
      if (!res.ok || !res.body) throw new Error("fetch failed");
      const total = Math.max(+res.headers.get("content-length") || 0, DATA_BYTES);
      const reader = res.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        onProgress(Math.min(got / total, 0.99));
      }
      const buf = new Uint8Array(got);
      let off = 0;
      for (const c of chunks) { buf.set(c, off); off += c.length; }
      const text = new TextDecoder().decode(buf);
      return JSON.parse(text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1));
    } catch (e) {
      return new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "data/app_data_v2.js";
        s.onload = () => resolve(window.VEO_DATA);
        s.onerror = reject;
        document.head.appendChild(s);
      });
    }
  }

  // Decode one scene: photo pixels, aligned truth and prediction in metres, |log ratio| and the crop mask.
  async function prepScene(sc) {
    if (sc._prep) return sc._prep;
    const img = await loadImg(sc.images.rgb);
    const c = mkCanvas(W, H);
    const cx = c.getContext("2d", { willReadFrequently: true });
    cx.drawImage(img, 0, 0, W, H);
    const rgb = cx.getImageData(0, 0, W, H).data;
    const gt = decodeDepth(sc.raw_depth.gt_b64_uint16);
    const pd = decodeDepth(sc.raw_depth.pred_b64_uint16);
    const lr = new Float32Array(W * H);
    const crop = new Uint8Array(W * H);
    let lo = Infinity, hi = -Infinity;
    for (let y = CROP.y0; y < CROP.y1; y++) {
      for (let x = CROP.x0; x < CROP.x1; x++) {
        const i = y * W + x;
        const g = gt[i], p = pd[i];
        if (g > 0 && p > 0) {
          crop[i] = 1;
          lr[i] = Math.abs(Math.log(p / g));
          if (g < lo) lo = g; if (g > hi) hi = g;
          if (p < lo) lo = p; if (p > hi) hi = p;
        }
      }
    }
    sc._prep = { sc, rgb, gt, pd, lr, crop, lo, hi };
    return sc._prep;
  }

  function webglOK() {
    try { const c = document.createElement("canvas"); return !!(window.THREE && (c.getContext("webgl") || c.getContext("experimental-webgl"))); }
    catch (e) { return false; }
  }

  window.VEO = {
    W, H, CROP, CAM, clamp, ease, smooth, mkCanvas, loadImg,
    TURBO, INFERNO, INFERNO_C, INFERNO_GLSL, lutCss, decodeDepth, loadData, prepScene, webglOK,
  };
})();
