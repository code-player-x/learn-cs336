"""Regression checks for guide export and the local teaching/tooling entry points."""
from __future__ import annotations

from contextlib import redirect_stdout
from html.parser import HTMLParser
import io
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

import code
from code.basic_study import post_install_check as scratch
import generate_output as export


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.images, self.math, self.prose = [], [], [], [], []
        self._math_depth = 0
        self._code_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {'pre', 'code', 'script', 'style'}:
            self._code_depth += 1
        if attrs.get('id'):
            self.ids.append(attrs['id'])
        if tag == 'a':
            self.links.append(attrs.get('href', ''))
        if tag == 'img':
            self.images.append(attrs.get('src', ''))
        if 'arithmatex' in attrs.get('class', '').split():
            self.math.append('')
            self._math_depth = 1
        elif self._math_depth:
            self._math_depth += 1

    def handle_endtag(self, tag):
        if tag in {'pre', 'code', 'script', 'style'}:
            self._code_depth -= 1
        if self._math_depth:
            self._math_depth -= 1

    def handle_data(self, data):
        if self._math_depth:
            self.math[-1] += data
        elif not self._code_depth:
            self.prose.append(data)


class ExportTests(unittest.TestCase):
    def test_math_is_preserved_and_code_is_not_math(self):
        source = '$x_1 + y_2$\n\n$$\nX \\in \\mathbb{R}^{n \\times d}\n$$\n\n`$literal$`\n'
        page = Page()
        page.feed(export.md_to_html(source))
        self.assertEqual(page.math, [r'\(x_1 + y_2\)', '\\[\nX \\in \\mathbb{R}^{n \\times d}\n\\]'])

    def test_nested_code_fences_render_as_code(self):
        rendered = export.md_to_html('1. example\n\n   ```python\n   x = 1\n   ```')
        self.assertIn('<pre>', rendered)
        self.assertIn('<code>', rendered)
        self.assertNotIn('<code>python', rendered)

    def test_list_display_formulas_render_without_raw_delimiters(self):
        page = Page()
        page.feed(export.md_to_html('- Dimensions:\n\n    $$\n    Q \\in \\mathbb{R}^{n \\times d}\n    $$\n'))
        self.assertEqual(len(page.math), 1)
        self.assertNotIn('$$', ''.join(page.prose))

    def test_links_images_and_heading_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            docs, interview, output = base / 'docs', base / 'interview', base / 'output'
            docs.mkdir()
            interview.mkdir()
            a, b = docs / '01.md', interview / '01.md'
            source = '# 第一章\n\n[local](#第一章) [other](../interview/01.md) [heading](../interview/01.md#第二%20章) [web](https://example.com/x?q=1#y)\n\n![img](images/test.png)'
            page = Page()
            page.feed(export.md_to_html(source, source=a, output_dir=output, included={a, b}))
            self.assertIn('docs-01--第一章', page.ids)
            self.assertEqual(page.links, ['#docs-01--第一章', '#interview-01', '#interview-01--第二-章', 'https://example.com/x?q=1#y'])
            self.assertEqual(page.images, ['../docs/images/test.png'])

    def test_combined_markdown_rebases_without_touching_code(self):
        base = Path('/tmp/project-export-fixture')
        source = base / 'docs' / '01.md'
        md = '[other](02.md) ![pic](../interview/images/a.png)\n\n`[literal](02.md)`\n\n```text\n[code](02.md)\n```'
        actual = export.rebase_markdown(md, source, base / 'output')
        self.assertIn('[other](../docs/02.md)', actual)
        self.assertIn('![pic](../interview/images/a.png)', actual)
        self.assertIn('`[literal](02.md)`', actual)
        self.assertIn('```text\n[code](02.md)\n```', actual)

    def test_source_read_failure_is_not_silently_published(self):
        with self.assertRaises(FileNotFoundError):
            export.read_md_file(Path('/definitely-missing/cs336-source.md'))

    def test_knowledge_docs_are_included_and_cross_linked(self):
        files = export.knowledge_files()
        topic_files = export.topic_content_files()
        self.assertTrue(files)
        self.assertTrue(topic_files)
        self.assertEqual(files[0].name, '00-总索引.md')
        for index, source in enumerate(files):
            self.assertTrue(source.name.startswith(f'{index:02d}-'), source.name)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            page = Page()
            page.feed(export.generate_html(output).read_text(encoding='utf-8'))
            self.assertIn('knowledge', page.ids)
            self.assertIn('topics', page.ids)
            for source in files:
                self.assertIn(export.section_id(source), page.ids)
            for source in topic_files:
                self.assertIn(export.section_id(source), page.ids)
            self.assertIn('#AI知识体系构建-00-总索引', page.links)
            self.assertIn('#AI知识体系构建-02-基础与模型原理', page.links)
            self.assertIn('#AI专题内容-07-RAG知识体系内容', page.links)
            self.assertIn('#AI专题内容-07-RAG知识体系内容--一rag-基础认知', page.links)
            self.assertIn('#AI知识体系构建-07-RAG知识体系', page.links)
            self.assertIn('#docs-03-Transformer架构详解', page.links)
            combined = export.generate_combined_markdown(output).read_text(encoding='utf-8')
            for source in files + topic_files:
                heading = source.read_text(encoding='utf-8').splitlines()[0]
                self.assertTrue(heading in combined, f'{source.name} missing from combined export')
            self.assertTrue('docs/AI' in combined, 'knowledge links were not rebased')

    def test_knowledge_math_has_no_unparsed_dollar_delimiters(self):
        formula_count = 0
        for source in export.knowledge_files() + export.topic_content_files():
            page = Page()
            page.feed(export.md_to_html(source.read_text(encoding='utf-8')))
            self.assertNotIn('$', ''.join(page.prose), f'{source.name}: unparsed math delimiter')
            formula_count += len(page.math)
        self.assertGreater(formula_count, 30)

    def test_rag_topic_keeps_nested_list_structure(self):
        source = export.DOCS_DIR / 'AI专题内容' / '07-RAG知识体系内容.md'
        rendered = export.md_to_html(source.read_text(encoding='utf-8'))
        self.assertIn('<li>核心定义<ul>', rendered)

    def test_full_project_export_has_no_broken_local_links_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            path = export.generate_html(output)
            page = Page()
            page.feed(path.read_text(encoding='utf-8'))
            self.assertGreater(len(page.math), 2000)
            self.assertNotIn('$$', ''.join(page.prose), 'A display formula was not parsed')
            self.assertIn('code', page.ids)
            self.assertIn('knowledge', page.ids)
            self.assertEqual(len(page.ids), len(set(page.ids)))
            for href in page.links + page.images:
                url = urlsplit(href)
                if url.scheme or url.netloc:
                    continue
                if href.startswith('#'):
                    self.assertIn(unquote(href[1:]), page.ids)
                elif url.path:
                    self.assertTrue((output / unquote(url.path)).exists(), href)


class ToolingTests(unittest.TestCase):
    def test_stdlib_console_api_survives_historical_package_name(self):
        console = code.InteractiveConsole()
        self.assertFalse(console.push('answer = 42'))
        self.assertEqual(console.locals['answer'], 42)
        self.assertIsNotNone(code.compile_command('answer = 42'))

    def test_scratch_encoder_respects_custom_pretokenization(self):
        vocab, merges = scratch.train_bpe('aa', 1)
        ids = scratch.bpe_encode('aa', merges, pattern=scratch.re.compile('.'))
        self.assertEqual(ids, [97, 97])
        self.assertEqual(scratch.bpe_decode(ids, vocab), 'aa')


if __name__ == '__main__':
    unittest.main()
