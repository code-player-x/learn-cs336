#!/usr/bin/env python3
"""
CS336 学习指南 - 多格式输出生成器
生成支持公式的 HTML 静态网站与合并 Markdown；PDF 可由浏览器打印保存。
"""

from __future__ import annotations

import argparse
import html
import os
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import markdown
from markdown.extensions import Extension
from markdown.extensions.toc import slugify_unicode
from markdown.treeprocessors import Treeprocessor
from pathlib import Path
from pygments.formatters import HtmlFormatter

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
INTERVIEW_DIR = PROJECT_ROOT / "interview"
OUTPUT_DIR = PROJECT_ROOT / "output"
COMICS_DIR = PROJECT_ROOT / "comics"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            if (typeof katex === 'undefined') {{
                document.getElementById('math-status').textContent = '公式资源未加载，请检查网络后刷新。';
                return;
            }}
            let failures = 0;
            document.querySelectorAll('.arithmatex').forEach(function (element) {{
                const tex = element.textContent;
                try {{
                    katex.render(tex.slice(2, -2), element, {{
                        displayMode: tex.startsWith('\\['), throwOnError: true, strict: 'ignore'
                    }});
                }} catch (error) {{
                    failures += 1;
                    element.classList.add('math-error');
                    element.title = error.message;
                }}
            }});
            document.getElementById('math-status').textContent = failures
                ? '有 ' + failures + ' 个公式无法渲染，请检查标红的原公式。'
                : '公式已渲染，可通过浏览器打印保存 PDF。';
        }});
    </script>
    <style>
        {highlight_css}
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
            color: white;
            border: 0;
            padding: 0;
            margin: 0 0 10px;
        }}
        nav .container {{ padding: 0; }}
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
        .arithmatex {{ overflow-x: auto; }}
        .math-error {{ color: #b71c1c; background: #ffebee; }}
        .katex-display {{ overflow-x: auto; overflow-y: hidden; padding: 4px 0; }}
        h1, h2, h3, h4, .content {{ scroll-margin-top: 150px; }}
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
        @media print {{
            nav, .toc, #math-status {{ display: none; }}
            body {{ background: white; }}
            .container {{ max-width: none; padding: 0; }}
            .content {{ box-shadow: none; padding: 0; }}
            pre {{ white-space: pre-wrap; overflow: visible; }}
            .katex-display {{ overflow: visible; }}
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
        <p id="math-status" role="status">正在加载公式资源……</p>
        {content}
    </div>
    <footer>
        <p>CS336 面试导向学习指南 | 基于 Stanford CS336: Language Modeling from Scratch</p>
        <p>本项目为非官方学习笔记；引用内容与依赖的版权、许可分别归原作者所有。</p>
    </footer>
</body>
</html>"""


def section_id(path: Path) -> str:
    """Stable, unique ids even when two directories contain the same filename."""
    return f"{path.parent.name}-{path.stem}"


def relative_destination(destination: str, source: Path, output_dir: Path) -> str:
    """Rebase local resources without changing external URLs, queries or fragments."""
    url = urlsplit(destination)
    if url.scheme or url.netloc or not url.path or url.path.startswith('/'):
        return destination
    target = (source.parent / unquote(url.path)).resolve()
    rebased = quote(os.path.relpath(target, Path(output_dir).resolve()), safe='/')
    return urlunsplit(('', '', rebased, url.query, url.fragment))


class _OutputLinks(Treeprocessor):
    def __init__(self, md, source: Path, output_dir: Path, included: set[Path]):
        super().__init__(md)
        self.source, self.output_dir, self.included = source, output_dir, included

    def run(self, root):
        prefix = section_id(self.source)
        for element in root.iter():
            if element.get('id'):
                element.set('id', f"{prefix}--{element.get('id')}")
            for attribute in ('href', 'src'):
                destination = element.get(attribute)
                if destination is None:
                    continue
                url = urlsplit(destination)
                if url.scheme or url.netloc or url.path.startswith('/'):
                    continue
                if attribute == 'href' and not url.path and url.fragment:
                    element.set(attribute, f"#{prefix}--{unquote(url.fragment)}")
                    continue
                target = (self.source.parent / unquote(url.path)).resolve() if url.path else None
                if attribute == 'href' and target in self.included:
                    anchor = section_id(target)
                    if url.fragment:
                        anchor += '--' + unquote(url.fragment)
                    element.set(attribute, '#' + anchor)
                else:
                    element.set(attribute, relative_destination(destination, self.source, self.output_dir))


class _OutputLinksExtension(Extension):
    def __init__(self, source: Path, output_dir: Path, included: set[Path]):
        self.source, self.output_dir, self.included = source, output_dir, included
        super().__init__()

    def extendMarkdown(self, md):
        # After inline parsing and TOC ids are assigned, before HTML serialization.
        md.treeprocessors.register(_OutputLinks(md, self.source, self.output_dir, self.included), 'output_links', 1)


def md_to_html(md_text: str, *, source: Path | None = None,
               output_dir: Path = OUTPUT_DIR, included: set[Path] | None = None) -> str:
    extensions = [
        "markdown.extensions.tables",
        "pymdownx.superfences",
        "pymdownx.highlight",
        "pymdownx.arithmatex",
        "markdown.extensions.toc",
    ]
    if source is not None:
        resolved_included = {path.resolve() for path in (included or set())}
        extensions.append(_OutputLinksExtension(source.resolve(), output_dir.resolve(), resolved_included))
    return markdown.markdown(md_text, extensions=extensions, extension_configs={
        'pymdownx.arithmatex': {'generic': True},
        'markdown.extensions.toc': {'slugify': slugify_unicode},
    })


def read_md_file(path: Path) -> str:
    # Do not silently publish a partial guide when a source cannot be read.
    return path.read_text(encoding="utf-8")


def generate_html(output_dir: Path = OUTPUT_DIR) -> Path:
    print("Generating HTML...")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = []

    sections.append('<div class="toc"><h2>目录</h2><ul>')
    sections.append('<li><strong>课程文档</strong></li>')

    doc_files = sorted(DOCS_DIR.glob("*.md"))
    interview_files = sorted(INTERVIEW_DIR.glob("*.md"))
    included = {f.resolve() for f in doc_files + interview_files}
    for f in doc_files:
        name = f.stem
        sections.append(f'<li><a href="#{html.escape(section_id(f))}">{html.escape(name)}</a></li>')

    sections.append('<li><strong>面试专区</strong></li>')
    for f in interview_files:
        name = f.stem
        sections.append(f'<li><a href="#{html.escape(section_id(f))}">{html.escape(name)}</a></li>')

    sections.append("</ul></div>")

    sections.append('<h1 id="lessons">课程文档</h1>')
    for f in doc_files:
        md_text = read_md_file(f)
        html_content = md_to_html(md_text, source=f, output_dir=output_dir, included=included)
        sections.append(
            f'<div class="content" id="{html.escape(section_id(f))}">{html_content}</div>'
        )

    sections.append('<h1 id="interview">面试专区</h1>')
    for f in interview_files:
        md_text = read_md_file(f)
        html_content = md_to_html(md_text, source=f, output_dir=output_dir, included=included)
        sections.append(
            f'<div class="content" id="{html.escape(section_id(f))}">{html_content}</div>'
        )

    sections.append('<h1 id="code">代码实现</h1><div class="content"><ul>')
    for folder in ('tokenizer', 'model', 'training', 'systems', 'alignment', 'basic_study'):
        for source in sorted((PROJECT_ROOT / 'code' / folder).glob('*.py')):
            if source.name == '__init__.py':
                continue
            destination = quote(os.path.relpath(source, output_dir), safe='/')
            label = source.relative_to(PROJECT_ROOT).as_posix()
            sections.append(f'<li><a href="{destination}">{html.escape(label)}</a></li>')
    sections.append('</ul></div>')
    sections.append('<h1 id="comics">漫画图解</h1>')
    sections.append('<div class="content">')
    comic_files = sorted(COMICS_DIR.glob("*.png"))
    for f in comic_files:
        rel_path = quote(os.path.relpath(f, output_dir), safe='/')
        sections.append(
            f'<h3>{html.escape(f.stem)}</h3><img src="{rel_path}" alt="{html.escape(f.stem)}">'
        )
    sections.append("</div>")

    full_content = "\n".join(sections)
    html_output = HTML_TEMPLATE.format(
        title="CS336 面试导向学习指南", content=full_content,
        highlight_css=HtmlFormatter().get_style_defs('.highlight')
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html_output, encoding="utf-8")
    print(f"HTML generated: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    return output_path


def rebase_markdown(md_text: str, source: Path, output_dir: Path) -> str:
    """Rebase ordinary Markdown links/images, preserving inline and fenced code."""
    protected = re.compile(r'^([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\2[ \t]*$|`+[^`\n]*`+', re.M)
    links = re.compile(r'(!?\[[^\]\n]*\]\()([^\s)]+)(\))')
    def rewrite(text):
        return links.sub(lambda m: m[1] + relative_destination(m[2], source, output_dir) + m[3], text)
    pieces, cursor = [], 0
    for match in protected.finditer(md_text):
        pieces.extend((rewrite(md_text[cursor:match.start()]), match[0]))
        cursor = match.end()
    pieces.append(rewrite(md_text[cursor:]))
    return ''.join(pieces)


def generate_combined_markdown(output_dir: Path = OUTPUT_DIR) -> Path:
    """Generate a single combined markdown for PDF conversion."""
    print("Generating combined markdown...")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    parts.append("# CS336 面试导向学习指南\n\n")
    parts.append("> Stanford CS336: Language Modeling from Scratch - 面试导向完整学习项目\n\n")
    parts.append("---\n\n")

    doc_files = sorted(DOCS_DIR.glob("*.md"))
    for f in doc_files:
        parts.append(rebase_markdown(read_md_file(f), f, output_dir))
        parts.append("\n\n---\n\n")

    parts.append("# 面试专区\n\n")
    interview_files = sorted(INTERVIEW_DIR.glob("*.md"))
    for f in interview_files:
        parts.append(rebase_markdown(read_md_file(f), f, output_dir))
        parts.append("\n\n---\n\n")

    combined_path = output_dir / "cs336-guide-combined.md"
    combined_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Combined markdown: {combined_path}")
    print(f"  Size: {combined_path.stat().st_size / 1024:.1f} KB")
    return combined_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR, help='输出目录，默认项目 output/')
    args = parser.parse_args()
    print("=" * 60)
    print("CS336 学习指南 - 多格式输出生成器")
    print("=" * 60)

    html_path = generate_html(args.output_dir)
    combined_md = generate_combined_markdown(args.output_dir)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"  HTML: {html_path}")
    print(f"  Markdown: {combined_md}")
    print()
    print('PDF: 在浏览器中打开 HTML，等“公式已渲染”后，打印并保存为 PDF。')
    print('注意：公式 JS/CSS 使用固定版本 CDN，需要网络；直接用 WeasyPrint 不会执行公式 JS。')
    print("=" * 60)


if __name__ == "__main__":
    main()
