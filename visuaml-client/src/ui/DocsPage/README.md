# DocsPage Components

This directory contains all components for the VisuaML documentation page.

## Structure

```
DocsPage/
├── DocsPage.types.ts              # TypeScript type definitions
├── DocsPage.constants.ts          # TOC sections and config constants
├── DocsPage.tsx                   # Main docs page component
├── DocsContent.tsx                # Documentation content sections
├── DocsSection.tsx                # Reusable section wrapper
├── DocsTableOfContents.tsx        # TOC component (desktop & mobile)
├── DocsTableOfContents.types.ts   # TOC component types
├── useScrollSpy.ts                # Hook for tracking active section
├── components/                    # Reusable sub-components
│   ├── TOCItem.tsx               # Individual TOC item
│   └── index.ts                  # Component exports
└── index.ts                       # Public exports
```

## Components

### DocsPage
Main documentation page with layout, header, and content sections.

### DocsContent
Contains all documentation sections (Get Started, Features, Usage, etc.).

### DocsSection
Reusable section wrapper with consistent styling and scroll margin.

### DocsTableOfContents
Sticky table of contents for desktop, collapsible for mobile.

### useScrollSpy
Custom hook that tracks which section is currently in view.

## Usage

```tsx
import { DocsPage } from '../ui/DocsPage';

// Route: /docs
<Route path="/docs" element={<DocsPage />} />
```

## Design Principles

- **Clean & Minimal**: Matching landing page aesthetic
- **Type-Safe**: All types defined in separate files
- **Maintainable**: Content separated from components
- **Responsive**: Mobile-friendly TOC and layout
- **Accessible**: Proper scroll spy and navigation
