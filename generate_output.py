#!/usr/bin/env python3
"""
CS336 学习指南 - 多格式输出生成器
生成 HTML 静态网站和 PDF 文档
"""

import os
import glob
import markdown
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "docs"
INTERVIEW_DIR = PROJECT_ROOT / "interview"
OUTPUT_DIR = PROJECT_ROOT / "output"
COMICS_DIR = PROJECT_ROOT / "comics"

OUTPUT_DIR.mkdir(exist_ok=True)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, 
                         "Helvetica Neue", Arial, "Noto Sans SC", sans-serif;
            line-height: 1.8;
            color: #333;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
            padding: 20px;
        }}
        nav {{
            background: #1a1a2e;
            color: white;
            padding: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        nav h1 {{
            font-size: 1.4em;
            margin-bottom: 10px;
        }}
        nav a {{
            color: #64b5f6;
            text-decoration: none;
            margin-right: 15px;
            font-size: 0.9em;
        }}
        nav a:hover {{ color: #90caf9; text-decoration: underline; }}
        .content {{
            background: white;
            padding: 40px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #3f51b5; padding-bottom: 10px; margin: 30px 0 20px; }}
        h2 {{ color: #283593; margin: 25px 0 15px; }}
        h3 {{ color: #3949ab; margin: 20px 0 10px; }}
        h4 {{ color: #5c6bc0; margin: 15px 0 8px; }}
        pre {{
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 16px;
            overflow-x: auto;
            font-size: 0.9em;
            line-height: 1.5;
        }}
        code {{
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        pre code {{ background: none; padding: 0; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px 14px;
            text-align: left;
        }}
        th {{ background: #e8eaf6; font-weight: 600; }}
        tr:nth-child(even) {{ background: #fafafa; }}
        blockquote {{
            border-left: 4px solid #3f51b5;
            padding: 10px 20px;
            margin: 15px 0;
            background: #e8eaf6;
            border-radius: 0 4px 4px 0;
        }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; }}
        .toc {{
            background: #e8eaf6;
            padding: 20px 30px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .toc h2 {{ margin-top: 0; }}
        .toc ul {{ list-style: none; padding-left: 0; }}
        .toc li {{ padding: 4px 0; }}
        .toc a {{ color: #3f51b5; text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        footer {{
            text-align: center;
            padding: 30px;
            color: #888;
            font-size: 0.85em;
        }}
        @media (max-width: 768px) {{
            .content {{ padding: 20px; }}
            nav {{ padding: 15px; }}
        }}
    </style>
</head>
<body>
    <nav>
        <div class="container">
            <h1>CS336 面试导向学习指南</h1>
            <div>
                <a href="#lessons">课程文档</a>
                <a href="#interview">面试专区</a>
                <a href="#code">代码实现</a>
                <a href="#comics">漫画图解</a>
            </div>
        </div>
    </nav>
    <div class="container">
        {content}
    </div>
    <footer>
        <p>CS336 面试导向学习指南 | 基于 Stanford CS336: Language Modeling from Scratch</p>
        <p>本项目仅供学习参考，课程内容版权归 Stanford CS336 课程组所有</p>
    </footer>
</body>
</html>"""


def md_to_html(md_text: str) -> str:
    extensions = [
        "markdown.extensions.tables",
        "markdown.extensions.fenced_code",
        "markdown.extensions.codehilite",
        "markdown.extensions.toc",
    ]
    return markdown.markdown(md_text, extensions=extensions)


def read_md_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"*文件读取失败: {e}*"


def generate_html():
    print("Generating HTML...")

    sections = []

    sections.append('<div class="toc"><h2>目录</h2><ul>')
    sections.append('<li><strong>课程文档</strong></li>')

    doc_files = sorted(DOCS_DIR.glob("*.md"))
    for f in doc_files:
        name = f.stem
        sections.append(f'<li><a href="#{name}">{name}</a></li>')

    sections.append('<li><strong>面试专区</strong></li>')
    interview_files = sorted(INTERVIEW_DIR.glob("*.md"))
    for f in interview_files:
        name = f.stem
        sections.append(f'<li><a href="#{name}">{name}</a></li>')

    sections.append("</ul></div>")

    sections.append('<h1 id="lessons">课程文档</h1>')
    for f in doc_files:
        md_text = read_md_file(f)
        html_content = md_to_html(md_text)
        sections.append(
            f'<div class="content" id="{f.stem}">{html_content}</div>'
        )

    sections.append('<h1 id="interview">面试专区</h1>')
    for f in interview_files:
        md_text = read_md_file(f)
        html_content = md_to_html(md_text)
        sections.append(
            f'<div class="content" id="{f.stem}">{html_content}</div>'
        )

    sections.append('<h1 id="comics">漫画图解</h1>')
    sections.append('<div class="content">')
    comic_files = sorted(COMICS_DIR.glob("*.png"))
    for f in comic_files:
        rel_path = os.path.relpath(f, OUTPUT_DIR)
        sections.append(
            f'<h3>{f.stem}</h3><img src="{rel_path}" alt="{f.stem}">'
        )
    sections.append("</div>")

    full_content = "\n".join(sections)
    html_output = HTML_TEMPLATE.format(
        title="CS336 面试导向学习指南", content=full_content
    )

    output_path = OUTPUT_DIR / "index.html"
    output_path.write_text(html_output, encoding="utf-8")
    print(f"HTML generated: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")


def generate_combined_markdown():
    """Generate a single combined markdown for PDF conversion."""
    print("Generating combined markdown...")

    parts = []
    parts.append("# CS336 面试导向学习指南\n\n")
    parts.append("> Stanford CS336: Language Modeling from Scratch - 面试导向完整学习项目\n\n")
    parts.append("---\n\n")

    doc_files = sorted(DOCS_DIR.glob("*.md"))
    for f in doc_files:
        parts.append(read_md_file(f))
        parts.append("\n\n---\n\n")

    parts.append("# 面试专区\n\n")
    interview_files = sorted(INTERVIEW_DIR.glob("*.md"))
    for f in interview_files:
        parts.append(read_md_file(f))
        parts.append("\n\n---\n\n")

    combined_path = OUTPUT_DIR / "cs336-guide-combined.md"
    combined_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Combined markdown: {combined_path}")
    print(f"  Size: {combined_path.stat().st_size / 1024:.1f} KB")
    return combined_path


def main():
    print("=" * 60)
    print("CS336 学习指南 - 多格式输出生成器")
    print("=" * 60)

    generate_html()
    combined_md = generate_combined_markdown()

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"  HTML: {OUTPUT_DIR / 'index.html'}")
    print(f"  Markdown: {combined_md}")
    print()
    print("To generate PDF, install weasyprint and run:")
    print("  pip install weasyprint")
    print("  python -c \"")
    print("    from weasyprint import HTML")
    print("    HTML('output/index.html').write_pdf('output/cs336-guide.pdf')")
    print('  "')
    print("=" * 60)


if __name__ == "__main__":
    main()
