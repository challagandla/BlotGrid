"use strict";

const canvas = document.getElementById("roi-canvas");
const ctx = canvas.getContext("2d");
const statusNode = document.getElementById("roi-status");
const viewport = document.getElementById("canvas-viewport");

let args = null;
let image = null;
let rois = [];
let selectedId = null;
let interaction = null;
let lastPostedHeight = -1;

const MIN_SIZE = 3;
const HANDLE_NAMES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

function post(type, extra = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}

function postHeight() {
  // Measure the content, not documentElement.scrollHeight: scrollHeight can
  // never report less than the iframe's own viewport, so once Streamlit has
  // sized the frame the editor could only ever grow. Shrinking the blot panel
  // would leave a permanent band of blank space below it.
  const shell = document.querySelector(".editor-shell");
  const height = Math.ceil(shell.getBoundingClientRect().height + 4);
  // Streamlit starts every component iframe at height 0 and only resizes it
  // when we report. Never report 0, or the editor stays invisible with no
  // event left that would re-measure it.
  if (height <= 4 || height === lastPostedHeight) return;
  lastPostedHeight = height;
  post("streamlit:setFrameHeight", { height });
}

function setFrameHeight() {
  // Report straight away rather than only from requestAnimationFrame: rAF is
  // suspended while the page is hidden (background browser tab, or a Streamlit
  // tab the user has not opened yet), and a height reported as 0 never gets a
  // second chance. The deferred calls then catch layout that settles later.
  postHeight();
  setTimeout(postHeight, 0);
  setTimeout(postHeight, 120);
}

function copyRois(items) {
  return items.map((roi) => ({ ...roi }));
}

function imageWidth() {
  return args ? Math.max(1, Number(args.image_width)) : 1;
}

function imageHeight() {
  return args ? Math.max(1, Number(args.image_height)) : 1;
}

function resizeAllowed() {
  return !args || args.allow_resize !== false;
}

function displayScale() {
  const rect = canvas.getBoundingClientRect();
  return rect.width ? rect.width / imageWidth() : 1;
}

function resizeSurface() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const cssWidth = rect.width;
  const cssHeight = cssWidth * imageHeight() / imageWidth();
  const dpr = clamp(window.devicePixelRatio || 1, 1, 3);
  const backingWidth = Math.max(1, Math.round(cssWidth * dpr));
  const backingHeight = Math.max(1, Math.round(cssHeight * dpr));
  const cssHeightText = `${cssHeight}px`;

  if (canvas.style.height !== cssHeightText) canvas.style.height = cssHeightText;
  if (canvas.width !== backingWidth) canvas.width = backingWidth;
  if (canvas.height !== backingHeight) canvas.height = backingHeight;
}

function applyDisplaySize() {
  // display_width is the on-screen width in CSS pixels; absent means "fit the
  // panel". viewport_height caps how much vertical space the blot may claim so
  // that a tall image does not push the controls below it off the screen.
  const requested = args ? Number(args.display_width) : 0;
  canvas.style.width = requested > 0 ? `${Math.max(32, Math.round(requested))}px` : "100%";
  const height = args ? Number(args.viewport_height) : 0;
  viewport.style.height = height > 0 ? `${Math.round(height)}px` : "";
}

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function selectedIndex() {
  return rois.findIndex((roi) => roi.roi_id === selectedId);
}

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * imageWidth() / rect.width,
    y: (event.clientY - rect.top) * imageHeight() / rect.height,
    scale: rect.width / imageWidth(),
  };
}

function handlesFor(roi) {
  const x0 = roi.x;
  const x1 = roi.x + roi.w;
  const y0 = roi.y;
  const y1 = roi.y + roi.h;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  return {
    nw: [x0, y0], n: [cx, y0], ne: [x1, y0], e: [x1, cy],
    se: [x1, y1], s: [cx, y1], sw: [x0, y1], w: [x0, cy],
  };
}

function hitHandle(roi, point) {
  if (!resizeAllowed()) return null;
  const radius = 8 / Math.max(point.scale, 0.05);
  const handles = handlesFor(roi);
  for (const name of HANDLE_NAMES) {
    const [x, y] = handles[name];
    if (Math.abs(point.x - x) <= radius && Math.abs(point.y - y) <= radius) return name;
  }
  return null;
}

function contains(roi, point) {
  return point.x >= roi.x && point.x <= roi.x + roi.w && point.y >= roi.y && point.y <= roi.y + roi.h;
}

function hitRoi(point) {
  for (let i = rois.length - 1; i >= 0; i -= 1) {
    if (rois[i].use && contains(rois[i], point)) return i;
  }
  return -1;
}

function cursorForHandle(handle) {
  if (["nw", "se"].includes(handle)) return "nwse-resize";
  if (["ne", "sw"].includes(handle)) return "nesw-resize";
  if (["n", "s"].includes(handle)) return "ns-resize";
  return "ew-resize";
}

function zoomText() {
  return `${imageWidth()}×${imageHeight()} px at ${Math.round(displayScale() * 100)}%`;
}

function updateStatus() {
  const index = selectedIndex();
  if (index < 0) {
    statusNode.textContent = `Click a rectangle to select it. · ${zoomText()}`;
    return;
  }
  const roi = rois[index];
  const activeCount = rois.filter((item) => item.use).length;
  const review = roi.review_required ? " · REVIEW REQUIRED" : "";
  const confidence = Number.isFinite(Number(roi.confidence)) ? Number(roi.confidence).toFixed(2) : "—";
  statusNode.textContent = `${roi.target} · ${roi.lane} — ${roi.status || "manual"}${review} · confidence ${confidence} · X ${roi.x}, Y ${roi.y}, W ${roi.w}, H ${roi.h} · ${activeCount} active ROIs · ${zoomText()}`;
}

function draw() {
  if (!image || !image.complete) return;
  // The backing surface follows the browser's pixel density, while drawing
  // coordinates remain in original-image pixels. This keeps the blot faithful
  // and makes overlay edges smooth at every display size and browser zoom.
  const scale = displayScale();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(canvas.width / imageWidth(), 0, 0, canvas.height / imageHeight(), 0, 0);
  ctx.imageSmoothingEnabled = scale <= 1.05;
  ctx.drawImage(image, 0, 0, imageWidth(), imageHeight());

  for (const roi of rois) {
    const selected = roi.roi_id === selectedId;
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash(roi.use ? [] : [7 / Math.max(scale, 0.05), 5 / Math.max(scale, 0.05)]);
    ctx.globalAlpha = roi.use ? 1 : 0.82;

    // A dark under-stroke preserves contrast on pale blot backgrounds. The
    // coloured stroke above it is deliberately solid and at least 2 CSS px.
    ctx.strokeStyle = "rgba(3, 7, 18, 0.92)";
    ctx.lineWidth = (selected ? 5.5 : 4.5) / Math.max(scale, 0.05);
    ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);
    ctx.strokeStyle = selected ? "#22d3ee" : roi.color;
    ctx.lineWidth = (selected ? 3.0 : 2.2) / Math.max(scale, 0.05);
    ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);

    const labelPrefix = !roi.use ? "EXCLUDED · " : (roi.review_required ? "REVIEW · " : "");
    const label = `${labelPrefix}${roi.target} · ${roi.lane}`;
    const fontSize = 11.5 / Math.max(scale, 0.05);
    ctx.font = `700 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
    const labelX = roi.x + 3 / scale;
    const labelY = Math.max(13 / scale, roi.y - 4 / scale);
    const labelWidth = ctx.measureText(label).width + 6 / scale;
    const labelHeight = 15 / scale;
    ctx.fillStyle = "rgba(3, 7, 18, 0.82)";
    ctx.fillRect(labelX - 3 / scale, labelY - 12 / scale, labelWidth, labelHeight);
    ctx.fillStyle = selected ? "#67e8f9" : roi.color;
    ctx.fillText(
      label,
      labelX,
      labelY,
    );
    ctx.restore();
  }

  const index = selectedIndex();
  if (index >= 0 && resizeAllowed()) {
    const handleRadius = clamp(4.5 / Math.max(scale, 0.05), 1.75, 14);
    ctx.save();
    for (const [x, y] of Object.values(handlesFor(rois[index]))) {
      ctx.beginPath();
      ctx.arc(x, y, handleRadius, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "rgba(3, 7, 18, 0.95)";
      ctx.lineWidth = 3 / Math.max(scale, 0.05);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, handleRadius, 0, Math.PI * 2);
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 1.6 / Math.max(scale, 0.05);
      ctx.stroke();
    }
    ctx.restore();
  }
  updateStatus();
}

function applySharedSizeFromSnapshot(width, height, activeIndex, activeRect, snapshot) {
  for (let i = 0; i < rois.length; i += 1) {
    if (i === activeIndex && activeRect) {
      rois[i] = { ...rois[i], ...activeRect, w: width, h: height };
      continue;
    }
    const cx = snapshot[i].x + snapshot[i].w / 2;
    const cy = snapshot[i].y + snapshot[i].h / 2;
    rois[i].w = width;
    rois[i].h = height;
    rois[i].x = Math.round(clamp(cx - width / 2, 0, imageWidth() - width));
    rois[i].y = Math.round(clamp(cy - height / 2, 0, imageHeight() - height));
  }
}

function resizedActive(start, handle, dx, dy) {
  let left = start.x;
  let right = start.x + start.w;
  let top = start.y;
  let bottom = start.y + start.h;
  if (handle.includes("w")) left = clamp(start.x + dx, 0, right - MIN_SIZE);
  if (handle.includes("e")) right = clamp(start.x + start.w + dx, left + MIN_SIZE, imageWidth());
  if (handle.includes("n")) top = clamp(start.y + dy, 0, bottom - MIN_SIZE);
  if (handle.includes("s")) bottom = clamp(start.y + start.h + dy, top + MIN_SIZE, imageHeight());
  return {
    x: Math.round(left), y: Math.round(top),
    w: Math.round(right - left), h: Math.round(bottom - top),
  };
}

function emitChange(action) {
  post("streamlit:setComponentValue", {
    value: {
      event_id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      action,
      selected_id: selectedId,
      rois: rois.map(({ color, ...roi }) => ({ ...roi })),
    },
  });
}

canvas.addEventListener("pointerdown", (event) => {
  const point = pointFromEvent(event);
  const currentIndex = selectedIndex();
  let handle = currentIndex >= 0 ? hitHandle(rois[currentIndex], point) : null;
  let index = currentIndex;
  if (!handle) {
    index = hitRoi(point);
    if (index >= 0) {
      selectedId = rois[index].roi_id;
      handle = hitHandle(rois[index], point);
    }
  }
  if (index < 0) {
    selectedId = null;
    draw();
    return;
  }
  interaction = {
    pointerId: event.pointerId,
    mode: handle ? "resize" : "move",
    handle,
    index,
    startPoint: point,
    snapshot: copyRois(rois),
  };
  canvas.setPointerCapture(event.pointerId);
  canvas.focus();
  draw();
  event.preventDefault();
});

canvas.addEventListener("pointermove", (event) => {
  const point = pointFromEvent(event);
  if (!interaction) {
    const index = selectedIndex();
    const handle = index >= 0 ? hitHandle(rois[index], point) : null;
    canvas.style.cursor = handle ? cursorForHandle(handle) : (hitRoi(point) >= 0 ? "move" : "default");
    return;
  }
  const dx = point.x - interaction.startPoint.x;
  const dy = point.y - interaction.startPoint.y;
  const start = interaction.snapshot[interaction.index];
  if (interaction.mode === "move") {
    rois[interaction.index].x = Math.round(clamp(start.x + dx, 0, imageWidth() - start.w));
    rois[interaction.index].y = Math.round(clamp(start.y + dy, 0, imageHeight() - start.h));
  } else {
    const active = resizedActive(start, interaction.handle, dx, dy);
    applySharedSizeFromSnapshot(active.w, active.h, interaction.index, active, interaction.snapshot);
  }
  draw();
  event.preventDefault();
});

function endPointer(event) {
  if (!interaction || event.pointerId !== interaction.pointerId) return;
  const action = interaction.mode;
  interaction = null;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  draw();
  emitChange(action);
}

canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);

canvas.addEventListener("keydown", (event) => {
  const index = selectedIndex();
  if (index < 0 || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
  const roi = rois[index];
  const step = event.shiftKey ? 10 : 1;
  if (event.altKey) {
    if (!resizeAllowed()) {
      event.preventDefault();
      return;
    }
    let width = roi.w;
    let height = roi.h;
    if (event.key === "ArrowLeft") width = Math.max(MIN_SIZE, width - step);
    if (event.key === "ArrowRight") width = Math.min(imageWidth(), width + step);
    if (event.key === "ArrowUp") height = Math.max(MIN_SIZE, height - step);
    if (event.key === "ArrowDown") height = Math.min(imageHeight(), height + step);
    applySharedSizeFromSnapshot(width, height, index, null, copyRois(rois));
    emitChange("resize-keyboard");
  } else {
    if (event.key === "ArrowLeft") roi.x = Math.max(0, roi.x - step);
    if (event.key === "ArrowRight") roi.x = Math.min(imageWidth() - roi.w, roi.x + step);
    if (event.key === "ArrowUp") roi.y = Math.max(0, roi.y - step);
    if (event.key === "ArrowDown") roi.y = Math.min(imageHeight() - roi.h, roi.y + step);
    emitChange("move-keyboard");
  }
  draw();
  event.preventDefault();
});

window.addEventListener("message", (event) => {
  if (!event.data || event.data.type !== "streamlit:render") return;
  args = event.data.args;
  const incoming = Array.isArray(args.rois) ? args.rois : [];
  rois = copyRois(incoming);
  const requestedSelectedId = args.selected_roi_id ? String(args.selected_roi_id) : "";
  if (requestedSelectedId && rois.some((roi) => roi.roi_id === requestedSelectedId)) {
    selectedId = requestedSelectedId;
  }
  if (selectedId && !rois.some((roi) => roi.roi_id === selectedId)) selectedId = null;
  applyDisplaySize();
  resizeSurface();
  image = new Image();
  image.onload = () => { draw(); setFrameHeight(); };
  image.src = args.image_url;
  setFrameHeight();
});

function handleResize() {
  // A manual panel resize or a browser-window change alters the on-screen
  // scale, so the overlay has to be redrawn at the new line weights.
  applyDisplaySize();
  resizeSurface();
  draw();
  setFrameHeight();
}

new ResizeObserver(handleResize).observe(document.querySelector(".editor-shell"));
new ResizeObserver(handleResize).observe(viewport);
window.addEventListener("resize", handleResize);
// A tab that mounts while hidden has no usable layout yet; re-measure when it
// becomes visible so the editor is never left at its collapsed height.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) handleResize();
});

post("streamlit:componentReady", { apiVersion: 1 });
setFrameHeight();
