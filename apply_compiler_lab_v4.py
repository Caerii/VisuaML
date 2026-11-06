#!/usr/bin/env python3
"""Apply the minimal integration edits after extracting the v4 overlay.

Idempotent, anchored substitutions; intentionally refuses to alter unknown
source layouts. Execute from the root of the VisuaML checkout, after extracting
this archive over it. Supports --check without writing.
"""
from __future__ import annotations

import argparse
from pathlib import Path


EDITS = {
    Path('visuaml-client/server/index.ts'): (
        ("import { finished } from 'stream/promises';",
         "import { finished } from 'stream/promises';\nimport { registerCompilerLabRoutes } from './compilerLab.ts';"),
        ('app.register(multipart);',
         'app.register(multipart);\nregisterCompilerLabRoutes(app);'),
    ),
    Path('visuaml-client/src/App.tsx'): (
        ("import { DocsPage } from './ui/DocsPage';",
         "import { DocsPage } from './ui/DocsPage';\nimport { CompilerLabPage } from './ui/CompilerLab/CompilerLabPage';"),
        ('        <Route path="/docs" element={<DocsPage />} />',
         '        <Route path="/docs" element={<DocsPage />} />\n        <Route path="/compiler" element={<CompilerLabPage />} />'),
    ),
    Path('visuaml-client/src/MainApp.tsx'): (
        ('        <TopBar />',
         '''        <TopBar />
        <a href="/compiler" style={{ position: 'absolute', zIndex: 20, right: 22, top: 92,
          padding: '8px 13px', borderRadius: 9, background: '#0c514d', color: '#e5fff8',
          textDecoration: 'none', fontSize: 13, fontWeight: 700 }}>
          Compiler Lab ↗
        </a>'''),
    ),
}


def apply(root: Path, check_only: bool = False) -> dict[str, str]:
    updated: dict[Path, str] = {}
    status: dict[str, str] = {}
    # First validate ALL anchor substitutions before modifying ANY file.
    for relative, edits in EDITS.items():
        filename = root / relative
        if not filename.is_file():
            raise FileNotFoundError(f'Required existing VisuaML file is missing: {relative}')
        text = filename.read_text(encoding='utf-8')
        changes = 0
        for anchor, replacement in edits:
            if replacement in text:
                continue  # Idempotence
            if text.count(anchor) != 1:
                raise ValueError(f'Expected exactly one insertion point {anchor!r} in {relative}')
            text = text.replace(anchor, replacement, 1)
            changes += 1
        updated[filename] = text
        status[str(relative)] = 'already applied' if changes == 0 else f'{changes} insertions'
    if not check_only:
        for filename, text in updated.items():
            filename.write_text(text, encoding='utf-8')
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, state in apply(args.root, args.check).items():
        print(f'{path}: {state}' + (' (check only)' if args.check else ''))


if __name__ == '__main__':
    main()
