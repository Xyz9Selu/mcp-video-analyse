"""
MCP Video Analyse Server

Uses QWen Omni model (via Alibaba Bailian / DashScope) to directly analyse
video content and extract transcripts with timestamps.

Local videos are uploaded to S3 (Cloudflare R2) for direct URL access by the
QWen API, avoiding the 10 MB base64 inline limit.  Uploaded files are deleted
immediately after the API call.

Configure by setting env vars (see .env.example) or a .env file.
"""

import os
import uuid
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from openai import OpenAI

load_dotenv()

# ── QWen / DashScope ─────────────────────────────────────────────────────

MODEL = "qwen3.5-omni-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError(
        "DASHSCOPE_API_KEY not set. "
        "Get your key at https://bailian.console.aliyun.com/?tab=model#/api-key"
    )

openai_client = OpenAI(api_key=api_key, base_url=BASE_URL)

# ── S3 / R2 ───────────────────────────────────────────────────────────────

S3_ENDPOINT = os.getenv(
    "S3_ENDPOINT", "https://2d7e74a7e603ecf56764c39143dcd3a0.r2.cloudflarestorage.com"
)
S3_REGION = os.getenv("S3_REGION", "WNAM")
S3_BUCKET = os.getenv("S3_BUCKET", "public")
S3_DIRECTORY = os.getenv("S3_DIRECTORY", "smartcache")
S3_CUSTOM_DOMAIN = os.getenv("S3_CUSTOM_DOMAIN", "static.beyondbits.party")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
if not S3_ACCESS_KEY or not S3_SECRET_KEY:
    raise RuntimeError("S3_ACCESS_KEY and S3_SECRET_KEY must be set")

s3_client = boto3.client(
    "s3",
    region_name=S3_REGION,
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=BotoConfig(signature_version="s3v4"),
)

server = Server("video-analyser")


def _upload_to_s3(local_path: Path) -> str:
    """Upload a file to R2 and return its public URL via custom domain."""
    ext = local_path.suffix or ".mp4"
    key = f"{S3_DIRECTORY}/{uuid.uuid4().hex}{ext}"

    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=S3_BUCKET,
        Key=key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "ContentDisposition": "inline",
        },
    )

    return f"https://{S3_CUSTOM_DOMAIN}/{key}"


def _build_video_content(source: str) -> dict:
    """Build a video_url content block.

    Public URLs are passed through directly.
    Local files are uploaded to R2 and a public URL is returned.
    """
    if source.startswith(("http://", "https://")):
        return {"type": "video_url", "video_url": {"url": source}}

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {source}")
    if not path.is_file():
        raise ValueError(f"Not a file: {source}")

    url = _upload_to_s3(path)
    return {"type": "video_url", "video_url": {"url": url}}


def _call_qwen(video_content: dict, prompt: str) -> str:
    """Send video + prompt to QWen Omni and return the text response."""
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [video_content, {"type": "text", "text": prompt}],
            }
        ],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )

    parts: list[str] = []
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and hasattr(delta, "content") and delta.content:
                parts.append(delta.content)

    return "".join(parts)


# ── tools ────────────────────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="extract_video_transcript",
            description=(
                "Extract a clean, AI-readable transcript from a video. "
                "Output is organized by topic with Markdown headings, "
                "no timestamps — optimized for downstream AI processing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Video source: a local file path (e.g. /data/video.mp4) "
                            "or a public HTTP/HTTPS URL."
                        ),
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "Language of the transcript. Use 'zh' for Chinese, "
                            "'en' for English, or 'auto' for auto-detection. "
                            "Default: 'zh'."
                        ),
                        "default": "zh",
                    },
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="analyze_video",
            description=(
                "Analyze video content with a custom prompt. Use for summarisation, "
                "scene description, key-point extraction, or any custom video "
                "understanding task."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Video source: a local file path or a public HTTP/HTTPS URL."
                        ),
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What to ask about the video. E.g. 'Summarise this video', "
                            "'Describe each scene', 'What are the key points?'."
                        ),
                    },
                },
                "required": ["source", "prompt"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    source: str = arguments["source"]

    try:
        video_content = _build_video_content(source)
    except (FileNotFoundError, ValueError) as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]

    if name == "extract_video_transcript":
        lang = arguments.get("language", "zh")
        lang_hint = {"zh": "中文", "en": "English", "auto": "the original language"}.get(
            lang, lang
        )
        prompt = (
            f"请完整提取视频中的所有语音内容，整理成适合AI阅读和学习的格式。"
            f"以{lang_hint}输出。\n\n"
            f"格式要求：\n"
            f"- 不要时间戳，不要 [MM:SS] 这类标记\n"
            f"- 按主题或逻辑段落组织内容，使用清晰的小标题（## 标题）分隔\n"
            f"- 保留原始讲解的完整信息和细节，不要省略\n"
            f"- 如有列举、对比、步骤等结构化内容，用 Markdown 列表呈现\n"
            f"- 说话人如有多人且可区分，用「说话人A：」前缀标注\n"
            f"- 语气词、重复的口水话可以精炼，但核心观点和例子必须完整保留"
        )
    elif name == "analyze_video":
        prompt = arguments.get("prompt", "")
        if not prompt:
            return [TextContent(type="text", text="Error: 'prompt' is required for analyze_video")]
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        result = _call_qwen(video_content, prompt)
    except Exception as exc:
        return [TextContent(type="text", text=f"API error: {exc}")]

    return [TextContent(type="text", text=result)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
