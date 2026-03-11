# Structured commit plan

Use this order from the repo root (`F:\Github\VisuaML.com`). Each block is one commit.

---

## 1. chore: move pnpm overrides to root and fix postinstall (workspace/install)

**Why first:** Unblocks install and applies to the whole workspace.

```powershell
git add package.json visuaml-client/package.json pnpm-lock.yaml
git add visuaml-client/scripts/postinstall.js
git commit -m "chore: move pnpm overrides to root, add picomatch, cross-platform postinstall"
```

---

## 2. refactor: split demo networks into modules and add generator

**Why second:** Core refactor and tooling; no UI yet.

```powershell
git add visuaml-client/src/lib/demoNetworks.ts
git add visuaml-client/src/lib/demoNetworks/
git add visuaml-client/scripts/generate-demo-networks.py
git add visuaml-client/scripts/README-DEMO-NETWORKS.md
git add visuaml-client/backend/requirements-demo.txt
git add visuaml-client/pyproject.toml
git add visuaml-client/src/ui/TopBar/TopBar.tsx
git add visuaml-client/src/ui/TopBar/useTopBar.ts
git commit -m "refactor: split demo networks into modules, add demo generator script

- demoNetworks/ with types, index, demos/*.ts
- Root demoNetworks.ts re-exports for backward compatibility
- generate-demo-networks.py and README, requirements-demo, pyproject
- TopBar: DemoNetwork type for demo options and export disable"
```

**Optional:** Add generated JSON in the same commit if you want them in git:

```powershell
git add visuaml-client/src/lib/demo-networks/
# then amend or include in the commit above
```

**Alternatively:** Ignore generated demos and document that users run `pnpm generate-demos-fast`:

- In `visuaml-client/.gitignore` add: `src/lib/demo-networks/*.json` (keep `.gitkeep` if present).

---

## 3. feat: add landing page, app shell, and routing

**Why third:** New app structure and entry UI.

```powershell
git add visuaml-client/index.html
git add visuaml-client/src/main.tsx
git add visuaml-client/src/App.tsx
git add visuaml-client/src/MainApp.tsx
git add visuaml-client/src/index.css
git add visuaml-client/src/pages/
git add visuaml-client/src/ui/LandingPage/
git add visuaml-client/src/ui/shared/
git commit -m "feat: add landing page, app shell, and routing

- MainApp with router, Landing and Docs pages
- Landing page sections and shared UI
- App.tsx/main.tsx/index.html and index.css updates"
```

---

## 4. feat: add documentation page

**Why fourth:** New page that depends on app shell.

```powershell
git add visuaml-client/src/ui/DocsPage/
git commit -m "feat: add documentation page with scroll spy and MUI Grid v7 layout"
```

---

## 5. docs: add analysis and future-direction docs

**Why last:** Documentation-only; optional to include.

```powershell
git add docs/LANDING_PAGE_ANALYSIS.md
git add docs/LANDING_PAGE_ANALYSIS_REALISTIC.md
git add docs/LANDING_PAGE_DIALECTIC_ANALYSIS.md
git add docs/future-directions/README.md
git add docs/future-directions/categorical-neural-architecture-search.md
git add visuaml-client/backend/docs-wip/CATGRAD_ANALYSIS_AND_PLAN.md
git commit -m "docs: add landing page analysis and categorical NAS / Catgrad docs"
```

---

## 6. chore: update .gitignore (optional)

Only if you want to ignore generated files or lockfiles:

```powershell
git add visuaml-client/.gitignore
git commit -m "chore: update .gitignore"
```

**Do not commit** (unless you have a reason):

- `visuaml-client/uv.lock` — add `visuaml-client/uv.lock` to `.gitignore` if you prefer not to track it.

---

## Summary

| # | Focus                    | Commit message (short) |
|---|--------------------------|-------------------------|
| 1 | Workspace / install      | chore: pnpm overrides, postinstall, picomatch |
| 2 | Demo networks + generator| refactor: demo networks modules and generator |
| 3 | Landing + app shell      | feat: landing page and routing |
| 4 | Docs page                | feat: documentation page |
| 5 | Docs / analysis          | docs: analysis and future directions |
| 6 | .gitignore (optional)    | chore: update .gitignore |

Run from repo root. After commits, push with `git push origin main` (or your branch).
