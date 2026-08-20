# opencode-picturereader

给纯文本模型用的本地图片理解工具链(opencode 插件)。无需网络、无需视觉模型,即可"看懂"图片:扫描布局、识别文字、取样材质、裁剪、调色板、对比、批量筛选,并可选接外部 VLM 做语义描述。

移植自 DeepSeek Harness 的 [picturereader](https://github.com/jing-hy/picturereader) 插件。

## 安装

把 `plugins/` 目录复制到全局或项目插件目录即可:

```bash
# 全局安装 (Windows)
copy /Y plugins\* %USERPROFILE%\.config\opencode\plugins\
# 或项目级安装
copy /Y plugins\* .opencode\plugins\
```

然后安装依赖(配置目录的 `package.json` 需包含这些包):

```bash
cd %USERPROFILE%\.config\opencode
npm install pngjs jpeg-js omggif
```

重启 opencode 后 9 个工具自动可用。

> 本地插件若要使用外部 npm 包,需要在配置目录的 `package.json` 里声明依赖(见 [opencode 插件文档](https://opencode.ai/docs/plugins#dependencies))。本仓库的 `package.json` 已声明,可复制合并。

## 工具

| 工具 | 作用 |
|---|---|
| `image_scan` | 粗粒度像素网格 + 颜色区域 + 结构提示,看布局/颜色/形状 |
| `image_ocr` | OCR 识别文字(windows / paddle / rapid 三引擎,默认 windows 内置) |
| `image_sample` | NxN 精确像素采样,判断材质/纹理 |
| `image_crop` | 按 0..1 区域裁剪成无损 PNG |
| `image_palette` | 主色 + 色相族分析 |
| `image_compare` | 两张图逐像素对比,输出差异率/差异框/verdict |
| `image_batch` | 批量清单(类型判断 + 是否有文字 + 摘要),用于大量图片的上下文筛选 |
| `vision_analyze` | 统一管线:低信息守卫 + 扫描 + OCR + 可选外部 VLM |
| `document_to_image` | 把 pdf/docx/xlsx/pptx 逐页转 PNG(需要 doc_venv + LibreOffice) |

## 配置(环境变量)

| 变量 | 默认 | 说明 |
|---|---|---|
| `PICTR_MODE` | `smart` | 路由模式:`privacy`(绝不外呼 VLM)/ `smart` / `strict` |
| `SEE_BASE` | GLM-4V 端点 | OpenAI 兼容视觉端点 URL |
| `SEE_MODEL` | `glm-4v-flash` | 视觉模型名 |
| `SEE_API_KEY` | - | 视觉 API key(也可用 `GLM_API_KEY`) |
| `SEE_MAX_TOKENS` | `8192` | VLM 最大输出 tokens |
| `SEE_TIMEOUT_MS` | `300000` | VLM 请求超时(ms) |
| `DSH_PADDLE_PYTHON` | `C:/Users/Administrator/paddle_venv/Scripts/python.exe` | PaddleOCR python(可选引擎) |
| `DSH_RAPID_PYTHON` | `C:/Users/Administrator/rapid_venv/Scripts/python.exe` | RapidOCR python(可选引擎) |
| `DSH_DOC_PYTHON` | `C:\Users\Administrator\doc_venv\Scripts\python.exe` | document_to_image 的 python |
| `DSH_SOFFICE` | LibreOffice 默认路径 | soffice.exe 路径(office 转 pdf) |

### 模式说明

- **privacy**:绝不调用任何外部 API,全部走本地工具(image_scan / image_ocr / image_sample)。
- **smart**(默认):先本地扫描,简单内容自己看,复杂内容值得时才调 VLM。
- **strict**:全证据 + 交叉验证。

### 外部 VLM(可选)

不配置也能用全部本地功能。要启用语义描述:

1. 注册 [智谱开放平台](https://open.bigmodel.cn) 获取 API Key;
2. 设置环境变量 `SEE_API_KEY=<你的key>`(默认端点即 GLM-4V-Flash);
3. `vision_analyze` 里传 `include_vlm: true` 即可。

## 隐私

`PICTR_MODE=privacy` 是硬门禁:即使配置了外部 API 也绝不外呼。所有图片只在本机处理。

## 许可证

MIT