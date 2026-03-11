# LandingPage Components

This directory contains all components for the VisuaML landing page.

## Structure

```
LandingPage/
├── LandingPage.types.ts          # TypeScript type definitions
├── LandingPage.constants.ts      # Section data and constants
├── LandingPage.styles.ts         # Shared style constants
├── LandingHeader.tsx             # Sticky header with logo and navigation
├── LandingHero.tsx               # Hero section with main heading and CTAs
├── LandingSections.tsx           # Main content sections
├── components/                   # Reusable sub-components
│   ├── SectionWithAnimation.tsx # Animated section wrapper
│   ├── SectionButton.tsx        # Reusable button with underline
│   ├── SectionDetailList.tsx    # Detail list component
│   └── index.ts                 # Component exports
└── index.ts                      # Public exports
```

## Components

### LandingHeader
Sticky header component with logo and navigation buttons.

### LandingHero
Full-viewport hero section with main heading, description, and CTA buttons.

### LandingSections
Scrollable content sections with animations and detail lists.

### Reusable Components

- **SectionWithAnimation**: Wraps sections with fade-in animation
- **SectionButton**: Text button with underline border effect
- **SectionDetailList**: Displays labeled detail items in a structured format

## Usage

```tsx
import { LandingPage } from '../pages/LandingPage';

// LandingPage internally uses:
// - LandingHeader
// - LandingHero
// - LandingSections
```

## Design Principles

- **Clean & Minimal**: Following superintelligent.group aesthetic
- **Type-Safe**: All types defined in separate files
- **Maintainable**: Constants extracted, components reusable
- **Responsive**: Mobile-first with proper breakpoints
- **Accessible**: Proper semantic HTML and ARIA where needed
