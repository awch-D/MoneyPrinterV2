# MoneyPrinterV2 使用说明

本文说明如何配置、运行两种视频能力，以及如何单独验证生图接口。

## 目录

- [环境要求](#环境要求)
- [安装与配置](#安装与配置)
- [命令行入口](#命令行入口)
- [能力一：短剧 / 话题短视频（short）](#能力一短剧--话题短视频short)
- [能力二：小说一章一集（novel_chapter）](#能力二小说一章一集novel_chapter)
- [横竖屏与成片尺寸](#横竖屏与成片尺寸)
- [生图接口说明与 curl 自测](#生图接口说明与-curl-自测)
- [运行结果](#运行结果)
- [常见问题](#常见问题)
- [更多文档](#更多文档)

---

## 环境要求

- **Python**：建议 3.12（与 `README.md` 一致）
- **ffmpeg**：MoviePy 写 MP4 需要系统已安装
- **ImageMagick**：烧录字幕需要，`config.json` 中配置 `imagemagick_path`（macOS 常见为 `/opt/homebrew/bin/magick`）
- **脚本大模型**：OpenAI 兼容的 `/v1/chat/completions`（话题文案、小说分镜 JSON 等）
- **文生图**：OpenAI 兼容的 `/v1/images/generations`（Gemini 代理等），见下文
- **TTS**：`kitten` 本地或 `qwen3` Gradio 等，见 [Configuration.md](./Configuration.md)
- **字幕**：默认本地 OpenAI **`whisper` 命令行**（需系统已安装，如 `brew install openai-whisper`），或 AssemblyAI

可选自检：

```bash
python3 scripts/preflight_local.py
```

---

## 安装与配置

```bash
cd MoneyPrinterV2
cp config.example.json config.json
python3.12 -m venv .venv   # 或 venv/
source .venv/bin/activate
pip install -r requirements.txt
```

编辑 **`config.json`**（勿提交真实密钥）：

| 用途 | 主要键名 |
|------|-----------|
| 脚本 / 分镜 LLM | `script_api_base_url`, `script_api_key`, `script_api_model` |
| 生图 | `nanobanana2_api_base_url`, `nanobanana2_api_key`（或环境变量 `GEMINI_API_KEY`）, `nanobanana2_model` |
| 生图比例 / 超时 | `nanobanana2_aspect_ratio`（须为比例字符串如 `16:9`）, `nanobanana2_image_timeout_seconds`, `nanobanana2_image_max_retries` |
| 小说章集分段上限 | `novel_chapter_max_segments` |
| 其它 | TTS、Whisper、`threads` 等见 [Configuration.md](./Configuration.md) |

---

## 命令行入口

所有命令在**仓库根目录**执行，并保证能加载 `src` 下的模块：

```bash
source .venv/bin/activate
export PYTHONUNBUFFERED=1    # 推荐：日志实时输出
PYTHONPATH=src python src/main.py [选项]
```

查看全部参数：

```bash
PYTHONPATH=src python src/main.py --help
```

---

## 能力一：短剧 / 话题短视频（`short`）

**默认能力**。流程：选题或读文件 → 脚本 → 多图提示 → 生图 → 整段 TTS → 字幕 → BGM → 成片。

**必填（二选一）**

- `--niche "领域"`：用于自动生成话题；或  
- `--script-file path.txt`：直接以文本为旁白（**不调用**脚本/生图 API，画面为占位图）

**常用示例**

```bash
PYTHONPATH=src python src/main.py --niche "科技" --language "Chinese"
```

```bash
PYTHONPATH=src python src/main.py --niche "科技" --language "Chinese" --topic "为什么边缘 AI 会改变创作"
```

```bash
PYTHONPATH=src python src/main.py --script-file stories/some_narration.txt --language "Chinese"
```

---

## 能力二：小说一章一集（`novel_chapter`）

**一章文本文件 = 一集视频**。流程：LLM 输出分镜 JSON（画风/角色/分段旁白与 `image_prompt`）→ 每段一张图 → 每段单独 TTS → 按真实时长对齐画面 → 整轨字幕（Whisper/AssemblyAI）→ BGM → 成片。

**必填**

- `--capability novel_chapter`
- `--chapter-file path/to/chapter.txt`：单章纯文本

**推荐**

- `--language`：与旁白语言一致（如 `Chinese`）
- `--orientation landscape` 或 `portrait`：与成片、生图比例一致（默认横屏 `landscape`）
- `--topic`：可选，用于日志与输出 JSON 中的标题标签（默认用文件名）

**示例**

```bash
PYTHONPATH=src python src/main.py \
  --capability novel_chapter \
  --chapter-file stories/tianbao_short.txt \
  --language Chinese \
  --orientation landscape \
  --topic 天宝短章
```

设计细节与数据结构见 [NovelChapter.md](./NovelChapter.md)。

生图阶段终端会**逐张**输出进度，例如：`生图 3/12：请求中…`，完成后：`生图 3/12 已完成 → xxx.jpg`（`short` 能力下的多图生成同样如此）。

---

## 横竖屏与成片尺寸

- `--orientation landscape`：当次运行将 **`video_output_aspect`** 与 **`nanobanana2_aspect_ratio`** 一并设为 **16:9**（不写回磁盘上的 `config.json`，仅内存覆盖）。
- `--orientation portrait`：对应 **9:16**。

这样生图比例与最终画幅一致，减少裁切问题。

---

## 生图接口说明与 curl 自测

项目与 **OpenAI 兼容** 的图像接口一致：

- **URL**：`{nanobanana2_api_base_url}/v1/images/generations`
- **方法**：`POST`
- **头**：`Authorization: Bearer <API_KEY>`，`Content-Type: application/json`
- **体**：`model`, `prompt`, `n`, `size`, `response_format: "b64_json"`
- **`size`**：必须是**比例字符串**（如 `16:9`、`9:16`），不要用 `1920x1080` 这类像素写法，否则代理常返回 400。

**用 curl 单独测一张图**（请把 `YOUR_API_KEY` 和地址换成自己的，勿泄露密钥）：

```bash
curl -sS -m 300 -X POST "https://你的代理域名/v1/images/generations" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d '{
    "model": "gemini-3.1-flash-image",
    "prompt": "Minimal test: single red circle on white background.",
    "n": 1,
    "size": "16:9",
    "response_format": "b64_json"
  }' \
  -o .mp/curl_test.json
```

单张 4K 图常需 **30～120 秒** 甚至更久，`-m` 超时建议 **≥300**。

---

## 运行结果

命令结束前会在**标准输出**打印一段 **JSON**，主要包括：

- `video_path`：成片 MP4 绝对路径  
- `audio_path`：旁白音频路径  
- `subtitle_path`：字幕文件路径（若生成失败可能为 `null`）  
- `image_paths`：用到的图片路径列表  
- `topic`、`script`（或章集拼接后的全文旁白）

临时与产出文件默认在仓库下 **`.mp/`**（首次运行会自动创建）。

---

## 常见问题

1. **生图很慢或超时**  
   属正常现象（尤其 4K、多段分镜串行）。可提高 `nanobanana2_image_timeout_seconds`，并保证网络稳定；或减少 `novel_chapter_max_segments` / 换更快模型（需代理支持）。

2. **小说章集 JSON 解析失败**  
   已对接 `response_format: json_object`（若网关支持）与多次 repair；仍失败时请检查 `script_api_*` 是否指向兼容的 Chat Completions 网关。

3. **字幕烧录失败**  
   检查 `imagemagick_path` 与中文字体 `subtitle_font`（中文需支持 CJK 的 `.ttf/.ttc`）。

4. **模块找不到**  
   务必使用 `PYTHONPATH=src`，且在仓库根目录执行 `python src/main.py`。

---

## 更多文档

- [Configuration.md](./Configuration.md)：配置项详解与 CLI 说明摘要  
- [NovelChapter.md](./NovelChapter.md)：小说章集能力专页  
- [README.md](../README.md)：英文简版与安装入口  
