/**
 * picturereader — opencode plugin port of the DSH "picturereader" local
 * image-understanding toolchain.
 *
 * Provides local (no network) pixel/OCR tools so a text-only model can "see"
 * images, plus an optional external VLM bridge for semantic descriptions:
 *
 *   image_scan        — coarse pixel grid + color regions + structure hints
 *   image_ocr         — OCR (Windows built-in / Paddle / Rapid)
 *   image_sample      — NxN exact-pixel material/texture sampling
 *   image_crop        — crop to a 0..1 region, write a lossless PNG
 *   image_palette     — dominant colors + hue-family breakdown
 *   image_compare     — pixel-wise difference of two images
 *   image_batch       — batch manifest (type guess + has-text + excerpts)
 *   vision_analyze    — unified pipeline: guard + scan + ocr + optional VLM
 *   document_to_image — pdf/office → per-page PNG (needs doc_venv + LibreOffice)
 *
 * Mode routing (PICTR_MODE env, default "smart"):
 *   privacy  — never call any external VLM (hard gate)
 *   smart    — local tools first, VLM only when worth it
 *   strict   — full evidence + cross-validation
 *
 * @module picturereader
 */

import { tool } from "@opencode-ai/plugin";
import * as core from "./picturereader-lib/core.js";
import { isLowInformationImage } from "./picturereader-lib/guard.js";
import { readFile, stat, writeFile, mkdir } from "node:fs/promises";
import { extname, resolve as pathResolve, join, dirname } from "node:path";
import { tmpdir } from "node:os";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { randomBytes } from "node:crypto";

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

const BYTE_CAP = 50 * 1024 * 1024;
const MAX_PIXELS = 24_000_000;

const z = tool.schema;

/** Normalized mode from env (privacy / smart / strict), default smart. */
function currentMode() {
  const v = String(process.env.PICTR_MODE ?? "smart").trim();
  return v === "privacy" || v === "strict" ? v : "smart";
}

function isPrivacy() {
  return currentMode() === "privacy";
}

/** Resolve a possibly-relative path against the session directory. */
function resolvePath(p, ctx) {
  return pathResolve(ctx.directory ?? process.cwd(), p);
}

/** Read a file as a Buffer after size/type checks; returns abs path + buffer. */
async function readImageFile(filePath, ctx, toolName) {
  const abs = resolvePath(String(filePath ?? "").trim(), ctx);
  if (abs.length === 0) throw new Error(`${toolName}: file_path must be a non-empty string`);
  let info;
  try {
    info = await stat(abs);
  } catch {
    throw new Error(`${toolName}: cannot read "${abs}": file not found`);
  }
  if (!info.isFile()) throw new Error(`${toolName}: cannot read "${abs}": not a regular file`);
  if (info.size > BYTE_CAP) throw new Error(`${toolName}: file exceeds ${BYTE_CAP} bytes`);
  const buf = await readFile(abs);
  return { abs, buf, info };
}

/** Validate extension and decode; enforce the pixel cap. */
function decodeChecked(buf, ext, toolName, path) {
  if (core.UNSUPPORTED_EXTENSIONS.has(ext)) {
    throw new Error(`${toolName}: WebP is not supported yet — convert the file to PNG or JPEG first`);
  }
  if (!core.IMAGE_EXTENSIONS.has(ext)) {
    throw new Error(`${toolName}: unsupported image type "${ext}" (supported: PNG, JPEG, GIF, BMP)`);
  }
  const image = core.decodeImage(buf, ext);
  if (image.width * image.height > MAX_PIXELS) {
    throw new Error(
      `${toolName}: ${image.width}x${image.height} exceeds the ${MAX_PIXELS}-pixel decode limit — downscale or crop the file first`
    );
  }
  return image;
}

function parseBoundedInt(raw, fallback, min, max, label) {
  const n = raw === undefined ? fallback : Number(raw);
  if (!Number.isInteger(n) || n < min || n > max) {
    throw new Error(`${label} must be an integer between ${min} and ${max}`);
  }
  return n;
}

function parseThreshold(raw, fallback, label) {
  const n = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(n) || n < 0 || n > 1) {
    throw new Error(`${label} must be a number between 0 and 1`);
  }
  return n;
}

const round3 = (v) => Math.round(v * 1000) / 1000;

// ---------------------------------------------------------------------------
// VLM bridge (simplified: env-based OpenAI-compatible endpoint, no llama-server)
// ---------------------------------------------------------------------------

const GLM4V_BASE = "https://open.bigmodel.cn/api/paas/v4";
const GLM4V_MODEL = "glm-4v-flash";

function vlmConfig() {
  const base = String(process.env.SEE_BASE ?? "").trim() || GLM4V_BASE;
  const model = String(process.env.SEE_MODEL ?? "").trim() || GLM4V_MODEL;
  const apiKey = String(process.env.SEE_API_KEY ?? "").trim() || String(process.env.GLM_API_KEY ?? "").trim();
  return { base, model, apiKey, maxTokens: Number(process.env.SEE_MAX_TOKENS ?? 8192), requestTimeoutMs: Number(process.env.SEE_TIMEOUT_MS ?? 300_000) };
}

function isVlmConfigured() {
  if (isPrivacy()) return false;
  const cfg = vlmConfig();
  if (cfg.base.length === 0) return false;
  return cfg.apiKey.length > 0;
}

async function sendVisionRequest(config, images, prompt) {
  const content = [{ type: "text", text: prompt }];
  for (const img of images) {
    content.push({ type: "image_url", image_url: { url: `data:${img.mime};base64,${img.base64}` } });
  }
  const body = { model: config.model, stream: false, messages: [{ role: "user", content }], max_tokens: config.maxTokens };
  const headers = { "content-type": "application/json" };
  if (config.apiKey) headers.authorization = `Bearer ${config.apiKey}`;
  const base = String(config.base || "").trim().replace(/\/+$/, "");
  let endpoint;
  if (/\/chat\/completions$/.test(base) || /\/v\d+\/chat\/completions$/.test(base)) {
    endpoint = base;
  } else if (/\/v\d+$/.test(base)) {
    endpoint = `${base}/chat/completions`;
  } else {
    endpoint = `${base}/v1/chat/completions`;
  }
  const res = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(config.requestTimeoutMs),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`picturereader: VLM HTTP ${res.status}: ${text.slice(0, 300)}`);
  }
  const json = await res.json();
  const contentText = json?.choices?.[0]?.message?.content;
  if (typeof contentText !== "string" || contentText.length === 0) {
    throw new Error("picturereader: VLM returned empty content");
  }
  return contentText;
}

// ---------------------------------------------------------------------------
// mode policy text (routing)
// ---------------------------------------------------------------------------

function routePolicyText() {
  const mode = currentMode();
  if (mode === "privacy") {
    return (
      "【当前模式：隐私模式】绝不调用任何外部视觉 API。对每张图只能使用本地工具：image_scan（看布局/颜色/结构）、image_ocr（读文字）、image_sample（细看材质纹理）。请用这些本地工具自行理解图片内容。"
    );
  }
  if (mode === "smart") {
    return (
      "【当前模式：智能模式】先用 image_scan 快速看一眼图片。然后自行判断：" +
      "（1）以文字为主 → 用 image_ocr 读文字即可，不必调 VLM；" +
      "（2）普通图表/界面/简单内容 → 用 image_scan + image_sample 自己看即可，不必调 VLM；" +
      "（3）仅当图片内容复杂、需要语义理解（如照片、抽象画面）" +
      (isVlmConfigured() ? "且值得时，才调用 vision_analyze(include_vlm=true) 走外部 VLM" : "时才尝试 VLM，但当前未配置外部 VLM，尽量用本地工具") +
      "。目标是减少调用轮数与耗时。"
    );
  }
  return (
    "【当前模式：严谨模式】自行选择路线并追求可靠：先用 image_scan 了解整体，必要时用 image_ocr 读文字、image_sample 细看细节。对关键判断采用交叉验证：把 image_scan / image_ocr ( / 外部 VLM) 多种证据相互对照。需要语义理解且值得时可用 vision_analyze(include_vlm=true) 走外部 VLM。"
  );
}

// ---------------------------------------------------------------------------
// tools
// ---------------------------------------------------------------------------

function imageScanTool() {
  return tool({
    description:
      "Read a local image file as a coarse pixel grid (downscaled + color-quantized) so a text-only model can see layout, colors and rough shapes. " +
      "Use it to inspect charts, screenshots, diagrams, UI mockups or photos: report dominant colors with percentages, relative positions of regions, coarse structure and luminance patterns. " +
      "The result includes a luminance grid (rows top->bottom, columns left->right; \" \"=transparent, \".\" darkest, \"@\" brightest), a color grid for colorful images, a \"grid coords\" line giving the row/col range, and a regions list: connected color blobs with position, size, aspect and texture density. " +
      "To inspect details: scan the full image (default size 32), find the region of interest, then scan again with focus: [row0, col0, row1, col1] keeping the SAME size, or region: [x0, y0, x1, y1] (0..1 fractions) which works with any size. " +
      "For fine detail use px_per_cell (e.g. 4 = each cell shows a 4x4 pixel area) with a small region/focus. " +
      "palette sets color depth: auto (default), full (14 colors), basic (8), gray. " +
      "size = target cells on the longer side (8..64, default 32). mode auto picks the color grid when the image is colorful. " +
      "Limitation: no OCR and no fine detail — zoom into a region to inspect details.",
    args: {
      file_path: z.string().describe("Path to the image file (PNG/JPEG/GIF/BMP)."),
      size: z.number().int().min(8).max(64).optional().describe("Target cell count on the longer side (8..64, default 32). Mutually exclusive with px_per_cell."),
      px_per_cell: z.number().int().min(1).max(512).optional().describe("Requested source pixels per cell for fine detail; use with region/focus on a small area, mutually exclusive with size."),
      mode: z.enum(["auto", "ascii", "color"]).optional().describe("auto = color grid when colorful, else luminance grid (default); ascii = luminance only; color = include color grid."),
      palette: z.enum(["auto", "full", "basic", "gray"]).optional().describe("Color depth: auto (default), full, basic, gray."),
      region: z.array(z.number()).optional().describe("Optional [x0, y0, x1, y1] fractions in 0..1 to zoom into part of the image. Mutually exclusive with focus."),
      focus: z.array(z.number().int()).optional().describe("Zoom target as grid coordinates [row0, col0, row1, col1] (inclusive). Mutually exclusive with region."),
    },
    async execute(args, ctx) {
      const ext = extname(String(args.file_path ?? "")).toLowerCase();
      const { abs, buf } = await readImageFile(args.file_path, ctx, "image_scan");
      const image = decodeChecked(buf, ext, "image_scan", abs);

      const mode = args.mode === undefined ? "auto" : String(args.mode);
      if (!["auto", "ascii", "color"].includes(mode)) throw new Error("image_scan: mode must be one of 'auto', 'ascii', 'color'");
      const palette = core.resolvePaletteArgument(args.palette);
      if (args.region !== undefined && args.focus !== undefined) {
        throw new Error("image_scan: region and focus are mutually exclusive — pass only one");
      }
      let pxPerCell;
      if (args.px_per_cell !== undefined) {
        if (args.size !== undefined) throw new Error("image_scan: size and px_per_cell are mutually exclusive — pass only one");
        pxPerCell = args.px_per_cell;
      }
      const size = pxPerCell !== undefined ? 32 : parseBoundedInt(args.size, 32, 8, 64, "image_scan: size");

      let regionArray;
      let regionDisplay;
      if (args.focus !== undefined) {
        const fullGridHeight = Math.max(1, Math.round(size * (image.height / image.width)));
        regionArray = core.resolveFocus(args.focus, size, fullGridHeight);
        regionDisplay = `focus [${args.focus.map(String).join(",")}]`;
      } else if (args.region !== undefined) {
        regionArray = core.normalizeRegion(args.region);
        regionDisplay = regionArray.map((v) => Math.round(v * 1000) / 1000).join(",");
      } else {
        regionDisplay = "full";
      }

      const analysis = core.analyzeImage(image.data, image.width, image.height, { size, mode, region: regionArray, palette, pxPerCell });
      return { output: core.renderImageScan({ path: abs, width: image.width, height: image.height, region: regionDisplay, ...analysis }) };
    },
  });
}

function imageOcrTool() {
  return tool({
    description:
      "Recognize text in a local image. Three engines: engine=\"windows\" (default) uses the Windows built-in OCR (no install, good for printed/UI text); engine=\"paddle\" uses PaddleOCR via the local paddle_venv; engine=\"rapid\" uses RapidOCR via the local rapid_venv. " +
      "Use it together with image_scan: when the pixel grid shows dense, regular, high-contrast structure that looks like text, call image_ocr on that region. " +
      "Parameters: file_path (required), region: [x0,y0,x1,y1] (0..1) or focus: [row0,col0,row1,col1] to restrict recognition, language (optional BCP-47 tag like \"zh-Hans\" or \"en-US\", Windows engine only), engine (windows default, paddle, rapid). " +
      "The result lists each recognized line with its pixel bounding box (and confidence score for paddle).",
    args: {
      file_path: z.string().describe("Path to the image file (PNG/JPEG/GIF/BMP)."),
      region: z.array(z.number()).optional().describe("Optional [x0, y0, x1, y1] fractions in 0..1 to restrict recognition. Mutually exclusive with focus."),
      focus: z.array(z.number().int()).optional().describe("Optional [row0, col0, row1, col1] grid coordinates (inclusive). Mutually exclusive with region."),
      language: z.string().optional().describe("Optional BCP-47 language tag (e.g. \"zh-Hans\", \"en-US\"); Windows engine only."),
      engine: z.enum(["windows", "paddle", "rapid"]).optional().describe("windows (default) = Windows built-in OCR; paddle = PaddleOCR via local paddle_venv; rapid = RapidOCR via local rapid_venv."),
    },
    async execute(args, ctx) {
      const ext = extname(String(args.file_path ?? "")).toLowerCase();
      const { abs, buf } = await readImageFile(args.file_path, ctx, "image_ocr");
      const image = decodeChecked(buf, ext, "image_ocr", abs);
      if (args.region !== undefined && args.focus !== undefined) {
        throw new Error("image_ocr: region and focus are mutually exclusive — pass only one");
      }
      const engine = args.engine === undefined ? "windows" : String(args.engine);
      if (!["windows", "paddle", "rapid"].includes(engine)) {
        throw new Error("image_ocr: engine must be 'windows' (default) or 'paddle' or 'rapid'");
      }

      let regionArray;
      let regionDisplay;
      if (args.focus !== undefined) {
        const fullGridHeight = Math.max(1, Math.round(32 * (image.height / image.width)));
        regionArray = core.resolveFocus(args.focus, 32, fullGridHeight);
        regionDisplay = `focus [${args.focus.map(String).join(",")}]`;
      } else if (args.region !== undefined) {
        regionArray = core.normalizeRegion(args.region);
        regionDisplay = regionArray.map((v) => Math.round(v * 1000) / 1000).join(",");
      } else {
        regionDisplay = "full";
      }

      const OPTIONAL = {
        paddle: { available: () => core.paddleAvailable(), install: "set DSH_PADDLE_PYTHON and create paddle_venv (see picturereader README)" },
        rapid: { available: () => core.rapidAvailable(), install: "set DSH_RAPID_PYTHON and create rapid_venv (see picturereader README)" },
      };
      let effectiveEngine = engine;
      let note;
      const opt = OPTIONAL[engine];
      if (opt !== undefined && !(await opt.available())) {
        effectiveEngine = "windows";
        note = `${engine[0].toUpperCase()}${engine.slice(1)}OCR is not installed (engine="${engine}" requested) — fell back to Windows OCR. To install: ${opt.install}`;
      }
      let result;
      try {
        result = await core.ocrImage(buf, ext, { region: regionArray, language: args.language, engine: effectiveEngine });
      } catch (error) {
        if (opt !== undefined && effectiveEngine === engine) {
          effectiveEngine = "windows";
          note = `${engine[0].toUpperCase()}${engine.slice(1)}OCR failed (${error.message.slice(0, 140)}) — fell back to Windows OCR.`;
          result = await core.ocrImage(buf, ext, { region: regionArray, language: args.language, engine: "windows" });
        } else {
          throw error;
        }
      }
      return { output: core.renderOcr({ path: abs, width: result.width, height: result.height, region: regionDisplay, engine: effectiveEngine, ...(note !== undefined ? { note } : {}), lines: result.lines }) };
    },
  });
}

function imageSampleTool() {
  return tool({
    description:
      "Sample a small region of a local image as an NxN grid of EXACT pixels (one real pixel per cell, not an average) plus a local-contrast statistic. " +
      "Use it to judge MATERIAL or TEXTURE: smooth gradients (skin, sky, water), high-contrast stripes (metal, wood grain), periodic repeats (fabric), high-frequency noise (foliage). " +
      "Workflow: first use image_scan to locate the area, then call image_sample with a SMALL region (e.g. [x0,y0,x1,y1] fractions covering roughly 30-400 px per side) and optional size (2..16, default 8). The region must be at least `size` pixels in each direction.",
    args: {
      file_path: z.string().describe("Path to the image file (PNG/JPEG/GIF/BMP)."),
      region: z.array(z.number()).describe("Required [x0, y0, x1, y1] fractions in 0..1: the small area to sample. Must cover at least `size` pixels in each direction."),
      size: z.number().int().min(2).max(16).optional().describe("Sample grid side length (2..16, default 8)."),
    },
    async execute(args, ctx) {
      const ext = extname(String(args.file_path ?? "")).toLowerCase();
      const { abs, buf } = await readImageFile(args.file_path, ctx, "image_sample");
      const image = decodeChecked(buf, ext, "image_sample", abs);
      const size = parseBoundedInt(args.size, 8, 2, 16, "image_sample: size");
      const regionArray = core.normalizeRegion(args.region);
      const sample = core.samplePixels(image.data, image.width, image.height, regionArray, size);
      return {
        output: core.renderSample({
          path: abs,
          width: sample.width,
          height: sample.height,
          region: regionArray.map((v) => Math.round(v * 1000) / 1000).join(","),
          contrast: sample.contrast,
          distinct: sample.distinct,
          stepX: sample.stepX,
          stepY: sample.stepY,
          points: sample.points,
        }),
      };
    },
  });
}

function imageCropTool() {
  return tool({
    description:
      "Crop a local image to a rectangular fraction region and write the result as a lossless PNG file. " +
      "Parameters: file_path (required), region (required, [x0, y0, x1, y1] fractions in 0..1), out_path (optional — where to write the PNG; when empty a unique file is created under the system temp directory picturereader/). " +
      "Returns the written output path plus the cropped pixel dimensions. Use image_scan / image_ocr on the returned path to continue analyzing.",
    args: {
      file_path: z.string().describe("Path to the source image file (PNG/JPEG/GIF/BMP)."),
      region: z.array(z.number()).describe("Required [x0, y0, x1, y1] fractions in 0..1 to crop to. Must obey x1 > x0 and y1 > y0."),
      out_path: z.string().optional().describe("Optional output path for the cropped PNG. When empty, a unique file is written under the system temp directory (picturereader/)."),
    },
    async execute(args, ctx) {
      const ext = extname(String(args.file_path ?? "")).toLowerCase();
      const { abs, buf } = await readImageFile(args.file_path, ctx, "image_crop");
      const image = decodeChecked(buf, ext, "image_crop", abs);
      const region = core.normalizeRegion(args.region);
      const cropped = core.cropRgba(image.data, image.width, image.height, region);
      const pngBytes = core.encodePng(cropped.data, cropped.width, cropped.height);

      let outPath;
      let generated = false;
      let tempDir;
      const explicitOut = args.out_path === undefined || args.out_path === null || String(args.out_path).trim().length === 0
        ? null
        : resolvePath(String(args.out_path).trim(), ctx);
      if (explicitOut !== null) {
        outPath = explicitOut;
        await mkdir(dirname(outPath), { recursive: true });
      } else {
        tempDir = join(tmpdir(), "picturereader");
        outPath = join(tempDir, `crop-${Date.now()}-${randomBytes(4).toString("hex")}.png`);
        generated = true;
        await mkdir(tempDir, { recursive: true });
      }
      await writeFile(outPath, pngBytes);
      const lines = [`crop: ${abs} (${cropped.width}x${cropped.height}) -> ${outPath}`];
      if (generated) lines.push(`written to generated temp file under ${tempDir}`);
      return { output: lines.join("\n") };
    },
  });
}

function imagePaletteTool() {
  return tool({
    description:
      "Extract the dominant colors of a local image (or a region) using 3-bit/channel quantization, plus a hue-family breakdown for an overall tone read. " +
      "Parameters: file_path (required), region (optional [x0,y0,x1,y1] fractions), top (1..32, default 12), sample_step (optional sampling stride, default 1). " +
      "Each dominant color gives its hex, classified name, percent share, and RGB tuple. hue_families groups colors by hue family regardless of darkness. distinct reports coarse color diversity.",
    args: {
      file_path: z.string().describe("Path to the source image file (PNG/JPEG/GIF/BMP)."),
      region: z.array(z.number()).optional().describe("Optional [x0, y0, x1, y1] fractions in 0..1 to restrict the analysis to part of the image."),
      top: z.number().int().min(1).max(32).optional().describe("Number of dominant colors to return (1..32, default 12)."),
      sample_step: z.number().int().min(1).max(100000).optional().describe("Optional sampling stride in pixels (default 1 = every pixel)."),
    },
    async execute(args, ctx) {
      const toolName = "image_palette";
      const ext = extname(String(args.file_path ?? "")).toLowerCase();
      const { abs, buf } = await readImageFile(args.file_path, ctx, toolName);
      const image = decodeChecked(buf, ext, toolName, abs);
      const top = parseBoundedInt(args.top, 12, 1, 32, `${toolName}: top`);
      const sampleStep = parseBoundedInt(args.sample_step, 1, 1, 100000, `${toolName}: sample_step`);

      const BUCKET_SHIFT = 5;
      const hexOf = (v) => v.toString(16).padStart(2, "0");
      const region = args.region === undefined ? [0, 0, 1, 1] : core.normalizeRegion(args.region);
      const [rx0, ry0, rx1, ry1] = core.normalizeRegion(region);
      const px0 = Math.max(0, Math.floor(rx0 * image.width));
      const px1 = Math.min(image.width, Math.ceil(rx1 * image.width));
      const py0 = Math.max(0, Math.floor(ry0 * image.height));
      const py1 = Math.min(image.height, Math.ceil(ry1 * image.height));

      const topList = [];
      const hueCounts = new Map();
      const buckets = new Map();
      let total = 0;
      for (let y = py0; y < py1; y += sampleStep) {
        for (let x = px0; x < px1; x += sampleStep) {
          const p = (y * image.width + x) * 4;
          if (image.data[p + 3] < 128) continue;
          const r = image.data[p];
          const g = image.data[p + 1];
          const b = image.data[p + 2];
          const key = (r >> BUCKET_SHIFT) << 6 | (g >> BUCKET_SHIFT) << 3 | (b >> BUCKET_SHIFT);
          let bucket = buckets.get(key);
          if (bucket === undefined) {
            bucket = { r: 0, g: 0, b: 0, count: 0 };
            buckets.set(key, bucket);
          }
          bucket.r += r;
          bucket.g += g;
          bucket.b += b;
          bucket.count += 1;
          const fam = core.hueFamilyFor(r, g, b);
          hueCounts.set(fam, (hueCounts.get(fam) ?? 0) + 1);
          total += 1;
        }
      }
      if (total > 0) {
        for (const bucket of buckets.values()) {
          const ar = Math.round(bucket.r / bucket.count);
          const ag = Math.round(bucket.g / bucket.count);
          const ab = Math.round(bucket.b / bucket.count);
          topList.push({
            hex: `#${hexOf(ar)}${hexOf(ag)}${hexOf(ab)}`,
            name: core.classify(ar, ag, ab, "full").name,
            pct: Math.round((bucket.count / total) * 1000) / 10,
            rgb: { r: ar, g: ag, b: ab },
            count: bucket.count,
          });
        }
        topList.sort((a, b) => b.count - a.count);
        for (const item of topList) delete item.count;
      }
      const hueFamilies = [...hueCounts.entries()]
        .map(([family, count]) => ({ family, pct: Math.round((count / total) * 1000) / 10 }))
        .sort((a, b) => b.pct - a.pct);

      const lines = [`palette: ${abs} (${image.width}x${image.height}, region=${region.map((v) => Math.round(v * 1000) / 1000).join(",")})`];
      if (topList.length > 0) {
        lines.push(`dominant colors: ${topList.slice(0, top).map((c) => `${c.name} ${c.pct}% (${c.hex} rgb(${c.rgb.r},${c.rgb.g},${c.rgb.b}))`).join(", ")}`);
      } else {
        lines.push("dominant colors: (none found)");
      }
      if (hueFamilies.length > 0) {
        const colored = hueFamilies.filter((h) => h.family !== "achromatic");
        const achromatic = hueFamilies.find((h) => h.family === "achromatic");
        lines.push(`hue families: ${colored.map((h) => `${h.family} ${h.pct}%`).join(", ")}${achromatic ? `, achromatic ${achromatic.pct}%` : ""}`);
      }
      lines.push(`distinct quantization buckets: ${buckets.size}`);
      return { output: lines.join("\n") };
    },
  });
}

function imageCompareTool() {
  return tool({
    description:
      "Compare two local images pixel-by-pixel, optionally within the same 0..1 fraction region of both, and report how different they are. " +
      "Parameters: file_path_a / file_path_b (required), region (optional [x0,y0,x1,y1] fractions applied to both — when omitted the full images are compared, aligned to the smaller size if dimensions differ), max_diff_threshold (optional 0..1, default 0.05), downsample (optional 1..32 sampling stride, default 4), preview_path (optional — write a PNG that marks differing pixels red on top of image A). " +
      "Returns mean_diff, diff_ratio, max_diff, size_diff, diff_box, and a verdict: \"size-diff\" when dimensions differ, \"different\" when diff_ratio or mean_diff exceeds the threshold, otherwise \"same\".",
    args: {
      file_path_a: z.string().describe("Path to the first image file (PNG/JPEG/GIF/BMP)."),
      file_path_b: z.string().describe("Path to the second image file (PNG/JPEG/GIF/BMP)."),
      region: z.array(z.number()).optional().describe("Optional [x0, y0, x1, y1] fractions in 0..1 applied to both images."),
      max_diff_threshold: z.number().optional().describe("Optional 0..1 threshold (default 0.05) controlling the verdict."),
      downsample: z.number().int().min(1).max(32).optional().describe("Optional sampling stride in pixels (1..32, default 4)."),
      preview_path: z.string().optional().describe("Optional output path for a difference preview PNG. When omitted no preview is written."),
    },
    async execute(args, ctx) {
      const toolName = "image_compare";
      const extA = extname(String(args.file_path_a ?? "")).toLowerCase();
      const extB = extname(String(args.file_path_b ?? "")).toLowerCase();
      const { abs: absA, buf: bufA } = await readImageFile(args.file_path_a, ctx, toolName);
      const { abs: absB, buf: bufB } = await readImageFile(args.file_path_b, ctx, toolName);
      const imageA = decodeChecked(bufA, extA, toolName, absA);
      const imageB = decodeChecked(bufB, extB, toolName, absB);
      const maxDiffThreshold = parseThreshold(args.max_diff_threshold, 0.05, `${toolName}: max_diff_threshold`);
      const downsample = parseBoundedInt(args.downsample, 4, 1, 32, `${toolName}: downsample`);

      const DIFF_PIXEL_THRESHOLD = 0.1;
      const sameSize = imageA.width === imageB.width && imageA.height === imageB.height;
      const sizeDiff = sameSize ? null : { w: Math.abs(imageA.width - imageB.width), h: Math.abs(imageA.height - imageB.height) };
      const region = args.region === undefined ? [0, 0, 1, 1] : core.normalizeRegion(args.region);
      const regionDisplay = region.map((v) => Math.round(v * 1000) / 1000).join(",");
      let note;
      if (!sameSize && args.region === undefined) {
        note = "images differ in size and no region was given — compared aligned whole images at the smaller dimensions";
      }

      const box = (imgW, imgH) => {
        const [rx0, ry0, rx1, ry1] = core.normalizeRegion(region);
        const x0 = Math.max(0, Math.floor(rx0 * imgW));
        const x1 = Math.min(imgW, Math.ceil(rx1 * imgW));
        const y0 = Math.max(0, Math.floor(ry0 * imgH));
        const y1 = Math.min(imgH, Math.ceil(ry1 * imgH));
        return { x0, y0, w: x1 - x0, h: y1 - y0 };
      };
      const boxA = box(imageA.width, imageA.height);
      const boxB = box(imageB.width, imageB.height);
      const gw = Math.min(boxA.w, boxB.w);
      const gh = Math.min(boxA.h, boxB.h);
      if (gw <= 0 || gh <= 0) throw new Error(`${toolName}: the comparison region has zero area`);

      let samples = 0;
      let diffPixels = 0;
      let meanSum = 0;
      let maxDiff = 0;
      let dMinX = Infinity, dMinY = Infinity, dMaxX = -1, dMaxY = -1;
      const cols = Math.max(1, Math.ceil(gw / downsample));
      const rows = Math.max(1, Math.ceil(gh / downsample));
      const cells = new Array(rows * cols);
      for (let gy = 0; gy < gh; gy += downsample) {
        const row = Math.floor(gy / downsample);
        for (let gx = 0; gx < gw; gx += downsample) {
          const col = Math.floor(gx / downsample);
          const ux = gw === 1 ? 0.5 : (gx + 0.5) / gw;
          const uy = gh === 1 ? 0.5 : (gy + 0.5) / gh;
          const ax = boxA.x0 + Math.floor(ux * boxA.w);
          const ay = boxA.y0 + Math.floor(uy * boxA.h);
          const bx = boxB.x0 + Math.floor(ux * boxB.w);
          const by = boxB.y0 + Math.floor(uy * boxB.h);
          const pa = (ay * imageA.width + ax) * 4;
          const pb = (by * imageB.width + bx) * 4;
          const dr = Math.abs(imageA.data[pa] - imageB.data[pb]) / 255;
          const dg = Math.abs(imageA.data[pa + 1] - imageB.data[pb + 1]) / 255;
          const db = Math.abs(imageA.data[pa + 2] - imageB.data[pb + 2]) / 255;
          const diff = (dr + dg + db) / 3;
          meanSum += diff;
          samples += 1;
          if (diff > maxDiff) maxDiff = diff;
          const differing = diff > DIFF_PIXEL_THRESHOLD;
          if (differing) {
            diffPixels += 1;
            if (ux < dMinX) dMinX = ux;
            if (uy < dMinY) dMinY = uy;
            if (ux > dMaxX) dMaxX = ux;
            if (uy > dMaxY) dMaxY = uy;
          }
          cells[row * cols + col] = differing ? [255, 0, 0] : [imageA.data[pa], imageA.data[pa + 1], imageA.data[pa + 2]];
        }
      }
      const meanDiff = samples === 0 ? 0 : meanSum / samples;
      const diffRatio = samples === 0 ? 0 : diffPixels / samples;
      const diffBox = dMinX === Infinity
        ? null
        : [round3(Math.min(dMinX, dMaxX)), round3(Math.min(dMinY, dMaxY)), round3(Math.max(dMinX, dMaxX)), round3(Math.max(dMinY, dMaxY))];

      let verdict;
      if (sizeDiff !== null) verdict = "size-diff";
      else if (diffRatio > maxDiffThreshold || meanDiff > maxDiffThreshold) verdict = "different";
      else verdict = "same";

      let preview;
      const explicitPrev = args.preview_path === undefined || args.preview_path === null || String(args.preview_path).trim().length === 0
        ? null
        : resolvePath(String(args.preview_path).trim(), ctx);
      if (explicitPrev !== null) {
        const rgba = Buffer.alloc(rows * cols * 4);
        for (let i = 0; i < cells.length; i += 1) {
          const [pr, pg, pb] = cells[i] ?? [0, 0, 0];
          rgba[i * 4] = pr;
          rgba[i * 4 + 1] = pg;
          rgba[i * 4 + 2] = pb;
          rgba[i * 4 + 3] = 255;
        }
        await mkdir(dirname(explicitPrev), { recursive: true });
        await writeFile(explicitPrev, core.encodePng(rgba, cols, rows));
        preview = explicitPrev;
      }

      const lines = [`compare: ${absA} (${imageA.width}x${imageA.height}) vs ${absB} (${imageB.width}x${imageB.height})`];
      lines.push(`verdict: ${verdict} | mean_diff=${round3(meanDiff)} diff_ratio=${round3(diffRatio)} max_diff=${round3(maxDiff)}`);
      if (sizeDiff) lines.push(`size_diff: w ${sizeDiff.w}, h ${sizeDiff.h}`);
      if (diffBox) lines.push(`difference region: ${diffBox.join(",")} (normalized within compared region)`);
      else lines.push("no differing pixels found (diff_box: null)");
      if (preview) lines.push(`preview: ${preview}`);
      if (note) lines.push(`note: ${note}`);
      return { output: lines.join("\n") };
    },
  });
}

function imageBatchTool() {
  return tool({
    description:
      "Batch-scale / context-validation tool: given a LIST of image paths, return one compact manifest (per-file type guess + whether it has text + a short scan/OCR excerpt + a recommendation) plus a whole-batch summary. " +
      "Use it when many images arrive together without individual instructions. It first probes a few images with OCR: if text-dense it treats the whole batch as documents and runs OCR on everything (auto_ocr=auto); otherwise it only OCRs the text-dense ones and triages the rest by pixel stats. " +
      "Each item reports: index, basename, width x height, type (text/table/photo/chart/blank/unknown), has_text, ocr_excerpt, scan_preview and a recommendation. Invalid files are recorded as errors, never a whole-batch failure.",
    args: {
      file_paths: z.array(z.string()).describe("List of image paths to batch (each may be relative to the working directory)."),
      auto_ocr: z.enum(["auto", "always", "never"]).optional().describe("auto (default) = probe the first few; if text-dense, run full OCR on all, else only on the text-dense ones. always = OCR every image. never = no OCR."),
      preview: z.enum(["scan", "none"]).optional().describe("scan (default) = include a truncated image_scan overview per image; none = skip scan previews."),
      max_files: z.number().int().min(1).optional().describe("Hard display cap on how many images to process in one call (default 50)."),
      probe_first: z.number().int().min(0).optional().describe("For auto_ocr='auto': how many leading images to OCR to decide whether the batch is text-dense (default 3)."),
      ocr_limit_chars: z.number().int().min(0).optional().describe("Max characters of OCR text to keep per image (default 800)."),
    },
    async execute(args, ctx) {
      const toolName = "image_batch";
      const rawPaths = args.file_paths;
      if (!Array.isArray(rawPaths) || rawPaths.length === 0) {
        throw new Error(`${toolName}: file_paths must be a non-empty array of image paths`);
      }
      const maxFiles = parseBoundedInt(args.max_files, 50, 1, 10000, `${toolName}: max_files`);
      const paths = rawPaths.map((p) => String(p).trim()).filter((p) => p.length > 0);
      if (paths.length === 0) throw new Error(`${toolName}: file_paths contains no usable paths`);
      if (paths.length > maxFiles) {
        throw new Error(`${toolName}: ${paths.length} files exceeds max_files=${maxFiles} for one call — split the batch and call again (or raise max_files)`);
      }
      const autoOcr = args.auto_ocr === undefined ? "auto" : String(args.auto_ocr);
      if (!["auto", "always", "never"].includes(autoOcr)) throw new Error(`${toolName}: auto_ocr must be one of 'auto', 'always', 'never'`);
      const previewMode = args.preview === undefined ? "scan" : String(args.preview);
      if (!["scan", "none"].includes(previewMode)) throw new Error(`${toolName}: preview must be 'scan' or 'none'`);
      const probeFirst = parseBoundedInt(args.probe_first, 3, 0, 10000, `${toolName}: probe_first`);
      const ocrLimitChars = parseBoundedInt(args.ocr_limit_chars, 800, 0, 100000, `${toolName}: ocr_limit_chars`);
      const scanSize = 16;
      const SOFT_OUTPUT_LIMIT = 6000;

      const nonEmptyLines = (ocrResult) => {
        const lines = ocrResult && Array.isArray(ocrResult.lines) ? ocrResult.lines : [];
        return lines.filter((l) => typeof l?.text === "string" && l.text.trim().length > 0).length;
      };
      const classifyType = (analysis, ocrLines, hasHorizontalStripes = false) => {
        const rough = analysis?.texture?.rough ?? 0;
        const shades = analysis?.distinctShades ?? 0;
        const hueFraction = (analysis?.hues ?? []).filter((h) => h.name !== "achromatic").reduce((s, h) => s + h.pct, 0) / 100;
        const regionCount = (analysis?.regions ?? []).length;
        if (ocrLines >= 2) return hasHorizontalStripes ? "table" : "text";
        if (ocrLines === 1 && hasHorizontalStripes) return "table";
        if (shades <= 1 && rough < 8) return "blank";
        if (regionCount >= 6 && hueFraction >= 0.05) return "chart";
        if (rough >= 15 || (shades >= 6 && hueFraction >= 0.15)) return "photo";
        if (hueFraction >= 0.2) return "photo";
        return "unknown";
      };
      const recommendFor = (type) => {
        switch (type) {
          case "text": return "read it with image_ocr (full text)";
          case "table": return "image_ocr for the cell text, then image_scan+sample for layout";
          case "chart": return "image_scan for axes/trends, then image_sample on points of interest";
          case "photo": return "rich photo-like content — the one case worth considering an external VLM";
          case "blank": return "skip — low information content";
          default: return "quick image_scan to confirm what it holds";
        }
      };
      const hasHorizontalStripes = (analysis) => (analysis?.structure ?? []).some((h) => /horizontal stripes/i.test(String(h)));
      const truncateScan = (rendered) => {
        const cut = 900;
        if (rendered.length <= cut) return rendered;
        const lines = rendered.split("\n");
        const kept = [];
        let total = 0;
        for (const line of lines) {
          if (total + line.length > cut) break;
          kept.push(line);
          total += line.length + 1;
        }
        const text = kept.join("\n");
        return text.length < rendered.length ? `${text}\n… (scan preview truncated)` : text;
      };

      const items = [];
      let processed = 0;
      let errors = 0;
      const decoded = [];

      for (let i = 0; i < paths.length; i += 1) {
        const filePath = paths[i];
        const entry = {
          index: i,
          path: filePath,
          basename: filePath.split(/[\\/]/).pop() || filePath,
          type: "unknown",
          has_text: false,
          recommendation: recommendFor("unknown"),
        };
        try {
          const ext = extname(filePath).toLowerCase();
          const { abs, buf } = await readImageFile(filePath, ctx, toolName);
          const image = decodeChecked(buf, ext, toolName, abs);
          entry.width = image.width;
          entry.height = image.height;
          entry.path = abs;
          processed += 1;
          decoded.push({ ...entry, image, raw: buf, ext });
        } catch (error) {
          entry.type = "unknown";
          entry.error = error.message;
          errors += 1;
          items.push(entry);
        }
      }

      if (decoded.length === 0) {
        const summary = `${toolName}: processed=0, errors=${errors} — none of the ${paths.length} file(s) could be decoded. Check the paths/extensions (PNG/JPEG/GIF/BMP) and file existence.`;
        return { output: summary };
      }

      const analyses = new Map();
      const analyze = (item) => {
        if (!analyses.has(item.index)) {
          analyses.set(item.index, core.analyzeImage(item.image.data, item.image.width, item.image.height, { size: scanSize, mode: "auto", region: undefined, palette: "auto", pxPerCell: undefined }));
        }
        return analyses.get(item.index);
      };

      let fullOcr = false;
      let ocrReason = null;
      const results = new Map();
      const runOcr = async (item) => {
        try {
          const res = await core.ocrImage(item.raw, item.ext, { engine: "windows" });
          results.set(item.index, { lines: res?.lines ?? [] });
        } catch (error) {
          results.set(item.index, { lines: [], note: `OCR failed (${error.message.slice(0, 120)})` });
        }
        return results.get(item.index);
      };

      if (autoOcr === "always") {
        fullOcr = true;
        ocrReason = "auto_ocr='always'";
      } else if (autoOcr === "never") {
        fullOcr = false;
        ocrReason = "auto_ocr='never' — no OCR run";
      } else {
        const probeCount = Math.min(probeFirst, decoded.length);
        let textDenseProbe = 0;
        for (let p = 0; p < probeCount; p += 1) {
          const res = await runOcr(decoded[p]);
          if (nonEmptyLines(res) >= 2) textDenseProbe += 1;
        }
        if (textDenseProbe > 0) {
          fullOcr = true;
          ocrReason = `probed first ${probeCount}; ${textDenseProbe} are text-dense -> treated the batch as documents and ran OCR on everything`;
        } else {
          fullOcr = false;
          ocrReason = `probed first ${probeCount}; none text-dense -> no full OCR (only individual text-dense images)`;
        }
      }

      for (const src of decoded) {
        const analysis = analyze(src);
        const structureHasHStripes = hasHorizontalStripes(analysis);
        let ocrLines = 0;
        let ocrText = "";
        let note = src.note;
        if (autoOcr === "never") {
          ocrLines = 0;
        } else if (fullOcr) {
          let res = results.get(src.index);
          if (res === undefined) res = await runOcr(src);
          ocrLines = nonEmptyLines(res);
          ocrText = (res?.lines ?? []).map((l) => l.text).filter(Boolean).join(" ");
          if (res?.note) note = note ? `${note}; ${res.note}` : res.note;
        } else {
          let res = results.get(src.index);
          if (res === undefined) res = await runOcr(src);
          ocrLines = nonEmptyLines(res);
          if (ocrLines >= 2) ocrText = (res?.lines ?? []).map((l) => l.text).filter(Boolean).join(" ");
          if (res?.note) note = note ? `${note}; ${res.note}` : res.note;
        }
        const type = classifyType(analysis, ocrLines, structureHasHStripes);
        const item = {
          index: src.index,
          path: src.path,
          basename: src.basename,
          width: src.width,
          height: src.height,
          type,
          has_text: ocrLines >= 2,
          recommendation: recommendFor(type),
          ...(note !== undefined ? { note } : {}),
        };
        if (ocrText.length > 0) {
          item.ocr_excerpt = ocrText.length > ocrLimitChars ? `${ocrText.slice(0, ocrLimitChars)}…` : ocrText;
        }
        if (previewMode === "scan") {
          item.scan_preview = truncateScan(core.renderImageScan({ path: src.path, width: src.width, height: src.height, region: "full", ...analysis }));
        }
        items.push(item);
      }

      const okItems = items.filter((it) => it.error === undefined || it.error === null);
      const textCount = okItems.filter((it) => it.type === "text" || it.type === "table").length;
      const photoCount = okItems.filter((it) => it.type === "photo").length;
      const blankCount = okItems.filter((it) => it.type === "blank").length;
      const scanCount = okItems.filter((it) => it.has_text).length;
      const bigText = okItems.filter((it) => it.type === "text" || it.type === "table").map((it) => it.index);
      const bigPhoto = okItems.filter((it) => it.type === "photo").map((it) => it.index);

      let summary =
        `${toolName}: ${processed} decoded / ${errors} error(s) out of ${paths.length} path(s). ` +
        `Types: ${textCount} text/table, ${photoCount} photo, ${blankCount} blank (rest mixed/unknown). ` +
        `${scanCount} image(s) contain text. ` +
        (fullOcr ? `Full OCR was run on the whole batch (${ocrReason}).` : `Full OCR was NOT run — ${ocrReason}.`) +
        ` Next step: ${bigPhoto.length > 0 ? `likely-photo indices worth a VLM look: ${bigPhoto.join(", ")}; ` : ""}` +
        `read the text-dense ones (${bigText.length > 0 ? bigText.join(", ") : "none"}) with image_ocr and scan the chart/table indices; skip the blank ones.`;

      const renderedTotal = JSON.stringify({ summary, items, processed, errors }).length;
      if (renderedTotal > SOFT_OUTPUT_LIMIT) {
        summary += " [TRUNCATED] The full manifest is large — instead of relying on these truncated excerpts, call image_ocr / image_scan directly on the specific indices above.";
      }

      const lines = [];
      lines.push(summary);
      for (const item of items) {
        if (item.error !== undefined) {
          lines.push(`  [!] ${item.index}. ${item.basename} — ERROR: ${item.error}`);
          continue;
        }
        let line = `[${item.index}] ${item.basename} ${item.width}x${item.height} type=${item.type}${item.has_text ? " text=yes" : ""} | ${item.recommendation}`;
        if (item.note) lines.push(`  note: ${item.note}`);
        if (item.ocr_excerpt) line += `\n    ocr: ${item.ocr_excerpt}`;
        if (item.scan_preview) line += `\n    scan: ${item.scan_preview}`;
        lines.push(line);
      }
      return { output: lines.join("\n") };
    },
  });
}

function visionAnalyzeTool() {
  return tool({
    description:
      "Unified image understanding: decode an image, run a low-information guard, optionally scan pixels, OCR text, and/or ask the VLM for a semantic description. " +
      "Use this when you need one call to both verify what is in the image and get a natural-language interpretation. " +
      "Returns evidence blocks: scan (pixel stats), ocr (real text), vlm (model description). If the low-information guard triggers and allow_low_info is false, VLM is not called. " +
      "VLM is optional: set SEE_BASE/SEE_MODEL/SEE_API_KEY (or GLM_API_KEY) to enable; in privacy mode (PICTR_MODE=privacy) VLM is never called. " +
      "Smart API calling: simple images skip VLM automatically. Multiple questions: call this tool multiple times with different prompts on the same image.",
    args: {
      file_path: z.string().describe("Path to the image file (PNG/JPEG/GIF/BMP)."),
      prompt: z.string().optional().describe("Question/instruction for the VLM, e.g. \"Describe this UI\"."),
      include_scan: z.boolean().optional().describe("Include pixel scan evidence (default true)."),
      include_ocr: z.boolean().optional().describe("Include OCR text evidence (default false; set true when text matters)."),
      ocr_engine: z.enum(["windows", "paddle", "rapid"]).optional().describe("OCR engine: windows (default), paddle or rapid."),
      include_vlm: z.boolean().optional().describe("Include VLM description (default true, but skipped if VLM not configured)."),
      allow_low_info: z.boolean().optional().describe("Skip the low-information guard and force VLM even on blank/simple images (default false)."),
    },
    async execute(args, ctx) {
      const toolName = "vision_analyze";
      const ext = extname(String(args.file_path ?? "")).toLowerCase();
      const { abs, buf } = await readImageFile(args.file_path, ctx, toolName);
      const image = decodeChecked(buf, ext, toolName, abs);

      const boolArg = (value, fallback = false) => {
        if (value === undefined || value === null) return fallback;
        if (typeof value === "boolean") return value;
        return String(value) === "true" || String(value) === "1";
      };
      const mode = currentMode();
      const privacy = isPrivacy();
      const defaults = privacy
        ? { includeScan: true, includeOcr: true, includeVlm: false, allowLowInfo: false }
        : mode === "strict"
          ? { includeScan: true, includeOcr: true, includeVlm: true, allowLowInfo: false }
          : { includeScan: true, includeOcr: false, includeVlm: true, allowLowInfo: false };
      const includeScan = args.include_scan === undefined ? defaults.includeScan : boolArg(args.include_scan, true);
      const includeOcr = args.include_ocr === undefined ? defaults.includeOcr : boolArg(args.include_ocr, false);
      const includeVlm = privacy ? false : (args.include_vlm === undefined ? defaults.includeVlm : boolArg(args.include_vlm, true));
      const allowLowInfo = boolArg(args.allow_low_info, false);
      const prompt = args.prompt ?? "Describe this image in detail.";
      const modePolicy = routePolicyText();
      const vlmAvailable = privacy ? false : isVlmConfigured();
      const shouldCallVlm = includeVlm && vlmAvailable;

      const lowInfo = isLowInformationImage(image.data, image.width, image.height);
      const blocks = [];
      let ocrText = "";
      let scanText = "";
      let vlmText = "";

      if (lowInfo && !allowLowInfo) {
        const message =
          "[vision_analyze] 低信息量拦截：图片空白或内容极少，为避免 VLM 幻觉，未调用 VLM。" +
          "请检查截图是否空白/未渲染/窗口在屏幕外；如确需识别请设置 allow_low_info=true。";
        return { output: message };
      }

      if (includeScan) {
        const analysis = core.analyzeImage(image.data, image.width, image.height, { size: 32, mode: "auto", region: undefined, palette: "auto" });
        scanText = core.renderImageScan({ path: abs, width: image.width, height: image.height, region: "full", ...analysis });
        blocks.push(`[scan]\n${scanText}`);
      }

      if (includeOcr) {
        const engine = args.ocr_engine ?? "windows";
        const ocr = await core.ocrImage(buf, ext, { engine });
        ocrText = core.renderOcr({ path: abs, width: ocr.width, height: ocr.height, region: "full", engine: ocr.engine, lines: ocr.lines });
        blocks.push(`[ocr]\n${ocrText}`);
      }

      if (shouldCallVlm) {
        const config = vlmConfig();
        const base64 = buf.toString("base64");
        const mime = ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : ext === ".png" ? "image/png" : ext === ".gif" ? "image/gif" : "image/bmp";
        const safePrompt =
          prompt +
          "\n\n重要：只描述图中明确可见的内容。如果图中没有明显物体/文字/界面元素，请直接回答：画面空白或内容极少。不要推测、不要脑补不存在的角色/场景/文字。";
        vlmText = await sendVisionRequest(config, [{ mime, base64 }], safePrompt);
        blocks.push(`[vlm]\n${vlmText}`);
      } else if (includeVlm && !vlmAvailable) {
        blocks.push("[vlm] VLM 未配置（未设置 SEE_BASE/SEE_API_KEY 或 GLM_API_KEY）。\n" +
          "要使用免费的 GLM-4V-Flash 视觉模型：注册 https://open.bigmodel.cn 获取 API Key，然后设置环境变量 SEE_API_KEY=你的key（默认端点即 GLM-4V）。");
      }

      const combined = [modePolicy, ...blocks].join("\n\n---\n\n");
      return { output: combined };
    },
  });
}

function documentToImageTool() {
  const SCRIPT_PATH = join(dirname(fileURLToPath(import.meta.url)), "picturereader-lib", "scripts", "doc-to-image.py");
  const DOC_VENV_PY = process.env.DSH_DOC_PYTHON ?? "C:\\Users\\Administrator\\doc_venv\\Scripts\\python.exe";
  const SUPPORTED_EXTS = new Set([".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt"]);
  const MAX_INPUT_BYTES = 512 * 1024 * 1024;

  return tool({
    description:
      "Convert a local Office/PDF document (pdf / docx / doc / xlsx / xls / pptx / ppt) into a list of per-page PNG image paths, so the pages can then be inspected with image_scan / image_ocr / vision_analyze. Purely local (no network). " +
      "Parameters: file_path (required) or file_paths (array), out_dir (optional, defaults to a temp dir), dpi (72..300, default 150), max_pages (1..500, default 50). " +
      "Requires the doc_venv Python (pymupdf) and LibreOffice (soffice) — set DSH_DOC_PYTHON / DSH_SOFFICE if not at default locations. Returns per document: input, page_count, pages: [{index, path, width, height, bytes}], out_dir.",
    args: {
      file_path: z.string().optional().describe("Path to a single document (pdf/docx/doc/xlsx/xls/pptx/ppt). Use either this or file_paths, not both."),
      file_paths: z.array(z.string()).optional().describe("Array of document paths to convert in one call (batch)."),
      out_dir: z.string().optional().describe("Optional output directory for the generated PNGs. Defaults to a temp dir under the system temp."),
      dpi: z.number().int().min(72).max(300).optional().describe("Render resolution in DPI (72..300, default 150)."),
      max_pages: z.number().int().min(1).max(500).optional().describe("Maximum number of pages to render (1..500, default 50)."),
    },
    async execute(args, ctx) {
      const toolName = "document_to_image";
      const dpi = parseBoundedInt(args.dpi, 150, 72, 300, `${toolName}: dpi`);
      const maxPages = parseBoundedInt(args.max_pages, 50, 1, 500, `${toolName}: max_pages`);
      const fp = typeof args.file_path === "string" ? args.file_path.trim() : "";
      const fps = Array.isArray(args.file_paths) ? args.file_paths.filter((x) => typeof x === "string" && x.trim().length > 0).map((x) => x.trim()) : [];
      if (fp.length > 0 && fps.length > 0) {
        throw new Error(`${toolName}: 请只传 file_path（单个）或 file_paths（批量），不要同时传两者。`);
      }
      const targets = fp.length > 0 ? [fp] : fps;
      if (targets.length === 0) throw new Error(`${toolName}: 需要一个输入文件（file_path 或 file_paths）。`);

      if (targets.length === 0) throw new Error(`${toolName}: 需要一个输入文件（file_path 或 file_paths）。`);

      // Verify python env + script exist up front for a clear message.
      let pyInfo;
      try {
        pyInfo = await stat(DOC_VENV_PY);
      } catch {
        throw new Error(
          `${toolName}: 转换所需的 Python 环境缺失。请先创建 doc_venv 并安装 pymupdf（默认路径 ${DOC_VENV_PY}，可用 DSH_DOC_PYTHON 覆盖）。`
        );
      }
      if (!pyInfo.isFile()) {
        throw new Error(`${toolName}: 转换所需的 Python 环境缺失（${DOC_VENV_PY} 不是文件）。`);
      }

      const outDir = (args.out_dir !== undefined && String(args.out_dir).trim().length > 0)
        ? resolvePath(String(args.out_dir).trim(), ctx)
        : join(tmpdir(), "picturereader-doc", `${Date.now()}-${randomBytes(4).toString("hex")}`);
      await mkdir(outDir, { recursive: true });

      const documents = [];
      for (let i = 0; i < targets.length; i += 1) {
        const rawPath = targets[i];
        const abs = resolvePath(rawPath, ctx);
        const display = abs;
        const ext = extname(display).toLowerCase();
        if (!SUPPORTED_EXTS.has(ext)) {
          throw new Error(`${toolName}: 不支持的文件类型 "${ext}"（支持: pdf / docx / doc / xlsx / xls / pptx / ppt）: ${display}`);
        }
        let info;
        try {
          info = await stat(abs);
        } catch {
          throw new Error(`${toolName}: 找不到文件: ${display}`);
        }
        if (!info.isFile()) throw new Error(`${toolName}: 不是普通文件: ${display}`);
        if (info.size > MAX_INPUT_BYTES) throw new Error(`${toolName}: 文件超过 ${MAX_INPUT_BYTES} 字节`);

        const prefix = `page_${i + 1}`;
        const res = spawnSync(DOC_VENV_PY, [SCRIPT_PATH, abs, outDir, prefix, String(dpi), String(maxPages)], {
          encoding: "utf8",
          timeout: 120_000,
          maxBuffer: 64 * 1024 * 1024,
        });
        if (res.error) {
          if (res.error.code === "ENOENT") throw new Error(`${toolName}: 无法运行 python: ${DOC_VENV_PY}`);
          if (res.error.code === "ETIMEDOUT") throw new Error(`${toolName}: 转换超时（>120s），请检查文档是否损坏、过大，或降低 max_pages / dpi。`);
          throw new Error(`${toolName}: 调用转换脚本失败: ${res.error.message}`);
        }
        if (res.status !== 0 || !res.stdout) {
          const body = (res.stderr || res.stdout || "").trim();
          let msg = body;
          try {
            const parsed = JSON.parse(body.split("\n")[0]);
            if (parsed && parsed.error) msg = parsed.error;
          } catch { /* raw text */ }
          throw new Error(`${toolName}: 转换失败: ${msg || `退出码 ${res.status}`}`);
        }
        let summary;
        try {
          summary = JSON.parse(res.stdout.trim().split("\n").filter(Boolean).pop());
        } catch (e) {
          throw new Error(`${toolName}: 无法解析转换脚本输出: ${e.message}`);
        }
        if (summary.error) throw new Error(`${toolName}: ${summary.error}`);
        documents.push({
          input: abs.split(/[\\/]/).pop(),
          page_count: summary.page_count ?? 0,
          rendered: (summary.pages || []).length,
          truncated: !!summary.truncated,
          pages: (summary.pages || []).map((p) => ({ index: p.index, path: p.path, width: p.width, height: p.height, bytes: p.bytes })),
        });
      }

      const totalPages = documents.reduce((s, d) => s + d.rendered, 0);
      const truncatedAny = documents.some((d) => d.truncated);
      const lines = [`documents converted to images (out_dir: ${outDir})`];
      for (const d of documents) {
        lines.push(`  ${d.input}: ${d.rendered}/${d.page_count} page(s) rendered${d.truncated ? " (truncated)" : ""}`);
        for (const p of d.pages) {
          lines.push(`    page ${p.index}: ${p.width}x${p.height}px, ${p.bytes} bytes → ${p.path}`);
        }
      }
      lines.push(`转换完成：${documents.length} 个文档，共渲染 ${totalPages} 页 PNG，输出目录 ${outDir}。` + (truncatedAny ? " 部分文档超过 max_pages 仅渲染前 N 页。" : ""));
      return { output: lines.join("\n") };
    },
  });
}

// ---------------------------------------------------------------------------
// plugin entry
// ---------------------------------------------------------------------------

export const picturereader = async () => ({
  tool: {
    image_scan: imageScanTool(),
    image_ocr: imageOcrTool(),
    image_sample: imageSampleTool(),
    image_crop: imageCropTool(),
    image_palette: imagePaletteTool(),
    image_compare: imageCompareTool(),
    image_batch: imageBatchTool(),
    vision_analyze: visionAnalyzeTool(),
    document_to_image: documentToImageTool(),
  },
});

export const server = picturereader;