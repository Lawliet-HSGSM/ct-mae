"use strict";
const data = JSON.parse(document.getElementById("dataset").textContent);
const byId = id => document.getElementById(id);
const number = (value, digits = 0) => Number(value).toLocaleString("vi-VN", {
  minimumFractionDigits: digits, maximumFractionDigits: digits
});
const memory = bytes => `${number(bytes / 1048576, 2)} MiB`;
const escapeHTML = value => String(value).replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[character]);
const baseline = data.resolutions[String(data.sizes[0])].splits;
const total = Object.values(baseline).reduce((sum, split) => sum + split.samples, 0);

byId("facts").innerHTML = [
  [number(total), "ảnh crop / một bản dữ liệu"],
  ["11", "lớp supervised"],
  ["1 kênh", "ảnh xám · uint8"],
  [data.sizes.join(" / "), "cạnh ảnh đã tải và kiểm tra (px)"]
].map(([value, label]) => `<div class="fact"><strong>${value}</strong><span>${label}</span></div>`).join("");

byId("class-select").innerHTML = data.metadata.labels.map((label, id) =>
  `<option value="${id}">${escapeHTML(data.labels_vi[id])} / ${escapeHTML(label)}</option>`
).join("");
byId("class-select").value = "6";

function filteredExamples() {
  const label = Number(byId("class-select").value);
  return data.examples[byId("split-select").value].filter(example => example.label === label);
}

function updateSampleOptions() {
  byId("sample-select").innerHTML = filteredExamples().map((example, position) =>
    `<option value="${position}">Mẫu ${position + 1} · index ${number(example.index)}</option>`
  ).join("");
  renderImages();
}

function histogramSVG(histogram) {
  const maximum = Math.max(...histogram, 1);
  const bars = histogram.map((count, value) => {
    const height = count / maximum * 44;
    return `<rect x="${value}" y="${48 - height}" width="1" height="${height}" fill="currentColor"/>`;
  }).join("");
  return `<svg class="histogram" viewBox="0 0 256 48" preserveAspectRatio="none"
    role="img" aria-label="Histogram pixel của ảnh: 0 bên trái, 255 bên phải">${bars}</svg>`;
}

function renderImages() {
  const examples = filteredExamples();
  const example = examples[Number(byId("sample-select").value)];
  byId("previous").disabled = byId("next").disabled = examples.length < 2;
  if (!example) {
    byId("image-grid").textContent = "Không có mẫu cho lựa chọn này.";
    byId("selection").textContent = "Không có mẫu.";
    return;
  }
  const split = byId("split-select").value;
  const organ = `${data.labels_vi[example.label]} / ${data.metadata.labels[example.label]}`;
  byId("selection").textContent = `${organ} · ${split} index ${number(example.index)} · label ${example.label}`;
  const native = byId("view-select").value === "native";
  byId("image-grid").innerHTML = data.sizes.map(size => {
    const image = example.images[String(size)];
    const stats = image.stats;
    const displaySize = native ? size : 224;
    return `<article class="image-card">
      <h3>${size} × ${size} <span class="secondary">· ${number(size * size)} pixels</span></h3>
      <div class="image-frame"><img src="${image.png}" data-resolution="${size}"
        alt="${escapeHTML(organ)}, ${split} index ${example.index}, bản ${size} pixel"
        width="${displaySize}" height="${displaySize}" style="width:${displaySize}px"></div>
      <div class="image-info">
        <p><strong>${size}×${size} · 1 kênh · ${stats.dtype}</strong></p>
        <p>Min–max: ${stats.min}–${stats.max} · mean ${number(stats.mean, 2)}</p>
        <p>Std: ${number(stats.std, 2)} · pixel 0: ${number(stats.zero_fraction * 100, 1)}%</p>
        <p class="secondary">Dữ liệu ảnh: ${number(stats.bytes / 1024, 2)} KiB<br>
          ${native ? "Hiển thị 1 pixel dữ liệu / 1 CSS pixel" : `Khung 224 px · phóng đại ${number(224 / size, 2)}×`}</p>
        ${histogramSVG(stats.histogram)}
        <p class="secondary">Histogram riêng của ảnh · 0 → 255</p>
      </div></article>`;
  }).join("");
}

byId("split-select").addEventListener("change", updateSampleOptions);
byId("class-select").addEventListener("change", updateSampleOptions);
byId("sample-select").addEventListener("change", renderImages);
byId("view-select").addEventListener("change", renderImages);
for (const [id, direction] of [["previous", -1], ["next", 1]]) {
  byId(id).addEventListener("click", () => {
    const length = filteredExamples().length;
    if (length) {
      byId("sample-select").value = String((Number(byId("sample-select").value) + direction + length) % length);
      renderImages();
    }
  });
}

byId("size-table").innerHTML = data.sizes.map(size => {
  const resolution = data.resolutions[String(size)];
  return `<tr><td><strong>${size}×${size}</strong></td><td>${number(size * size)}</td>
    <td>${number(size / 28, 2)}×</td><td>${number((size / 28) ** 2, 2)}×</td>
    <td>${number(size * size / 1024, 2)} KiB</td>
    <td>${number(size * size * 4 / 1024, 2)} KiB</td>
    <td>${memory(resolution.asset.bytes)}</td><td class="ok">Đã khớp</td></tr>`;
}).join("");
const notLoaded = Object.keys(data.metadata.assets).map(Number).filter(size => !data.sizes.includes(size));
byId("available-sizes").textContent = notLoaded.length
  ? `Bản chính thức chưa tải: ${notLoaded.map(size => `${size}×${size} (${memory(data.metadata.assets[String(size)].bytes)})`).join(", ")}. Chưa đo thống kê cho các bản này.`
  : "Đã tải và kiểm tra tất cả các độ phân giải chính thức.";

byId("stats-table").innerHTML = data.sizes.flatMap(size =>
  ["train", "val", "test"].map(split => {
    const stats = data.resolutions[String(size)].splits[split];
    return `<tr><td>${size} / ${split}</td><td>[${stats.shape.join(", ")}]</td>
      <td>${number(stats.labeled_samples)} / ${number(stats.samples)}</td><td>${stats.dtype}</td>
      <td>${stats.min}–${stats.max}</td><td>${number(stats.mean, 2)}</td><td>${number(stats.std, 2)}</td>
      <td>${number(stats.p01, 1)} / ${number(stats.p50, 1)} / ${number(stats.p99, 1)}</td>
      <td>${number(stats.zero_fraction * 100, 2)}%</td><td>${memory(stats.array_bytes)}</td></tr>`;
  })
).join("");

byId("class-table").innerHTML = data.metadata.labels.map((label, id) => {
  const fraction = baseline.train.class_counts[id] / baseline.train.samples * 100;
  return `<tr><td>${id}</td><td>${escapeHTML(data.labels_vi[id])} / ${escapeHTML(label)}</td>
    <td>${number(baseline.train.class_counts[id])}</td><td>${number(baseline.val.class_counts[id])}</td>
    <td>${number(baseline.test.class_counts[id])}</td>
    <td><span class="bar" aria-hidden="true"><i style="width:${fraction}%"></i></span>${number(fraction, 2)}%</td></tr>`;
}).join("");
const messages = [...data.warnings];
if (!data.label_alignment) messages.push("Chỉ tải một độ phân giải; chưa thể kiểm tra alignment giữa nhiều bản.");
byId("warnings").textContent = messages.join(" ");
byId("sample-note").textContent = `${data.samples_per_class} ảnh / lớp / split, seed ${data.seed}.`;
updateSampleOptions();
