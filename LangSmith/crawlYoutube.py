import datetime
import os
from typing import List, Optional
from dotenv import load_dotenv

# ========== 1. LangSmith — must be set BEFORE load_dotenv() ==========
# load_dotenv() can overwrite env vars set after it runs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from my_llm import embeddings

model = ChatDeepSeek(model_name="deepseek-chat")
persist_dir = "chroma_data_dir"

urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=gFSRIht95h0",
    "https://www.youtube.com/watch?v=DkNqgCz8cjE",
    "https://www.youtube.com/watch?v=657Agkgga44",
    "https://www.youtube.com/watch?v=c5yDkwjZG80",
    "https://www.youtube.com/watch?v=433SmtTc0TA",
    "https://www.youtube.com/watch?v=AZ6257Ya_70",
]

# ========== 2. Date parsing — shared, covers all loaders ==========
_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y%m%d",  # FIX Bug 3: yt-dlp returns '20231105'
]


def _parse_date(value) -> Optional[datetime.datetime]:
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str):
        for fmt in _DATE_FORMATS:
            try:
                return datetime.datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


# ========== 3. add_publish_year — mutate in-place, no rebind needed ==========
# FIX Bug 2: original code did `doc = add_publish_year(doc)` inside a for-loop,
# which rebinds the local variable but never updates the list element.
# The fix: mutate doc.metadata directly — no return value needed.
def add_publish_year(doc: Document) -> None:
    try:
        parsed = _parse_date(doc.metadata.get("publish_date"))
        doc.metadata["publish_year"] = parsed.year if parsed else None
        if parsed is None and doc.metadata.get("publish_date"):
            print(f"  警告: 无法解析日期 '{doc.metadata['publish_date']}'")
    except Exception as e:
        print(f"  警告: 添加发布年份失败 — {e}")
        doc.metadata["publish_year"] = None


# ========== 4. Primary loader — YoutubeLoader ==========
# FIX Bug 1: add_video_info=True triggers a pytube metadata fetch that YouTube
# now blocks with HTTP 400. Set it to False to load transcript only.
def load_youtube_videos(urls: List[str]) -> List[Document]:
    docs, failed = [], []
    print(f"开始加载 YouTube 视频 (共 {len(urls)} 个)...")
    print("=" * 50)
    for i, url in enumerate(urls, 1):
        print(f"正在加载 [{i}/{len(urls)}]: {url}")
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,  # ← FIX: avoids pytube HTTP 400
                language=["zh", "zh-Hans", "zh-Hant", "en"],
            )
            loaded = loader.load()
            docs.extend(loaded)
            print(f"  ✓ 成功 ({len(loaded)} 个文档)")
        except Exception as e:
            print(f"  ✗ 失败: {str(e)[:120]}")
            failed.append(url)

    print(f"\n总结: 成功 {len(docs)} 个，失败 {len(failed)} 个")
    if failed:
        print("失败 URL:")
        for u in failed:
            print(f"  - {u}")
    return docs


# ========== 5. Fallback loader — yt-dlp ==========
def load_youtube_videos_ytdlp(urls: List[str]) -> List[Document]:
    from yt_dlp import YoutubeDL

    docs, failed = [], []
    ydl_opts = {"quiet": True, "no_warnings": True}

    for i, url in enumerate(urls, 1):
        print(f"[yt-dlp] 正在加载 [{i}/{len(urls)}]: {url}")
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                content = "\n\n".join(filter(None, [
                    f"Title: {info.get('title', '')}",
                    f"Description: {info.get('description', '')}",
                ]))
                # yt-dlp 'upload_date' format: '20231105' — covered by _DATE_FORMATS
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "title": info.get("title", ""),
                        "author": info.get("uploader", ""),
                        "publish_date": info.get("upload_date", ""),  # '20231105'
                        "duration": info.get("duration", 0),
                        "views": info.get("view_count", 0),
                    },
                )
                docs.append(doc)
                print(f"  ✓ 成功: {info.get('title', '')[:60]}")
        except Exception as e:
            print(f"  ✗ 失败: {str(e)[:120]}")
            failed.append(url)

    print(f"\n[yt-dlp] 总结: 成功 {len(docs)} 个，失败 {len(failed)} 个")
    return docs


# ========== 6. Main ==========
def main():
    # Primary: YoutubeLoader
    docs = load_youtube_videos(urls)

    # Fallback: yt-dlp for any that failed
    if not docs:
        print("\n主加载器全部失败，切换到 yt-dlp 备用方案...")
        docs = load_youtube_videos_ytdlp(urls)

    if not docs:
        print("错误: 所有加载方式均失败，程序退出")
        return None

    print(f"\n预览第一个文档:")
    print(f"  元数据: {docs[0].metadata}")
    print(f"  内容 (前300字符): {docs[0].page_content[:300]}")

    # FIX Bug 2: mutate in-place — no `doc =` rebind needed
    print("\n添加视频发布年份...")
    for doc in docs:
        add_publish_year(doc)

    years = [doc.metadata["publish_year"] for doc in docs if doc.metadata.get("publish_year")]
    if years:
        print(f"年份范围: {min(years)} — {max(years)}")

    print("\n分割文档...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = splitter.split_documents(docs)
    print(f"分割完成: {len(split_docs)} 个块")

    print("\n构建向量数据库...")
    try:
        vectorstore = Chroma.from_documents(
            split_docs,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        count = vectorstore._collection.count()
        print(f"向量数据库已保存到: {persist_dir}  ({count} 个向量)")
    except Exception as e:
        print(f"错误: 构建向量数据库失败 — {e}")
        return None

    print("\n" + "=" * 50)
    print("处理完成!")
    return vectorstore


if __name__ == "__main__":
    vectorstore = main()