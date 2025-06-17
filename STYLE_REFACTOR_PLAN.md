# VisuaML Style Refactoring Plan

This document outlines the plan, progress, and process for migrating the VisuaML client from a single, monolithic `index.css` file to a modern, type-safe, and maintainable CSS-in-JS architecture using MUI's styling engine (Emotion).

## 1. The Problem

The initial styling was handled by a single `index.css` file exceeding 1600 lines. This approach led to several issues:
- **Lack of Scalability**: Difficult to manage and add new styles without conflicts.
- **No Type Safety**: Styling relied on raw strings for class names and CSS variables, making it prone to typos and silent errors.
- **Poor Maintainability**: Global scope made it hard to know which styles affected which components, making refactoring risky.
- **Technical Debt**: The file contained significant redundancy, especially in animation and theming definitions.

## 2. The Solution: A Gradual CSS-in-JS Migration

We are adopting a **gradual migration** strategy to move to a more robust system without a "big bang" rewrite that would halt development.

**The End Goal:**
- **Co-located Styles**: Each component will have its styles defined in a corresponding `.styles.ts` file.
- **Type-Safe Theming**: A single source of truth for design tokens (`colors`, `spacing`, `shadows`, etc.) will be defined in a TypeScript theme object.
- **Maintainability**: Removing the global CSS file will make the codebase easier to reason about and safer to change.

## 3. Current Progress & Completed Steps

We have successfully established the foundation for this migration and completed a full end-to-end test with a pilot component.

- ✅ **Dependencies Installed**: `@emotion/react` and `@emotion/styled` are added to the project.
- ✅ **Type-Safe Theme Created**: A centralized theme has been created at `visuaml-client/src/styles/theme.ts`. This file translates the original CSS variables into a typed MUI theme object.
- ✅ **Theme Provided**: The entire application in `main.tsx` is now wrapped with MUI's `<ThemeProvider>`, making the typed theme accessible to all components.
- ✅ **Pilot Migration Complete**: The `NetworkStatsDisplay` component has been fully migrated:
  - Its styles were converted to styled-components in `NetworkStatsDisplay.styles.ts`.
  - The component was refactored to use these new styles.
  - The old, corresponding `.network-stats-*` rules were deleted from `index.css`.

## 4. The Playbook: How to Migrate the Next Component

To continue the refactor, follow this safe, repeatable process for each remaining component styled by `index.css`.

**Step 1: Choose a Target Component**
- Select the next component to migrate (e.g., `TopBar`). Use `grep` or a search to find all its associated CSS classes in `index.css`.

**Step 2: Create a `.styles.ts` File**
- In the component's directory, create a new file (e.g., `TopBar.styles.ts`).
- Import `styled` from `@mui/material/styles` and any necessary MUI components you'll be styling (e.g., `Button`, `Select`).
- Translate the CSS rules from `index.css` into exported `styled()` components. Use the `theme` object for type-safe access to colors, spacing, shadows, etc.

**Step 3: Refactor the Component (`.tsx`)**
- Import all your new styled components: `import * as S from './Component.styles';`.
- Replace the JSX elements that use `className` attributes with their new styled-component counterparts (e.g., replace `<div className="topbar__container">` with `<S.Container>`).

**Step 4: Delete the Dead CSS**
- Once the component is fully refactored and working correctly, delete the old CSS rules from `index.css`.
- Leave a comment indicating that the styles have been migrated, for example: `/* STYLES FOR .topbar-* migrated to TopBar.styles.ts */`.

**Step 5: Verify**
- Ensure the application still looks and functions as expected.

## 5. Next Targets

The following is a suggested order for migrating the remaining components, from least to most complex:

1.  **`TopBar`**: A good next step. It's self-contained and has several distinct child elements.
2.  **Base Button Styles**: Refactor the `.neural-button` classes into a reusable, styled `Button` component that can be imported.
3.  **`MLNode` and its children**: This is the most complex part. It should be tackled last and potentially broken down further (e.g., migrate `NodeHeader` first, then `NodeDetailsPane`).

By following this plan, we can systematically and safely improve the codebase, one piece at a time. 