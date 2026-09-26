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
            source = '# 第一章\n\n[local](#第一章) [other](../interview/01.md) [web](https://example.com/x?q=1#y)\n\n![img](images/test.png)'
            page = Page()
            page.feed(export.md_to_html(source, source=a, output_dir=output, included={a, b}))
            self.assertIn('docs-01--第一章', page.ids)
            self.assertEqual(page.links, ['#docs-01--第一章', '#interview-01', 'https://example.com/x?q=1#y'])
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

    def test_full_project_export_has_no_broken_local_links_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            path = export.generate_html(output)
            page = Page()
            page.feed(path.read_text(encoding='utf-8'))
            self.assertGreater(len(page.math), 2000)
            self.assertNotIn('$$', ''.join(page.prose), 'A display formula was not parsed')
            self.assertIn('code', page.ids)
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
