/**
 * Constants for DocsPage
 */
import type { TOCSection } from './DocsPage.types';

export const DOCS_SECTIONS: TOCSection[] = [
  { id: 'get-started', label: 'Get Started' },
  { id: 'features', label: 'Features' },
  { id: 'usage', label: 'Usage' },
  { id: 'architecture', label: 'System Architecture' },
  { id: 'categorical-foundation', label: 'Categorical Foundation' },
  { id: 'bridge-architecture', label: 'Bridge Architecture' },
  { id: 'collaboration', label: 'Real-time Collaboration' },
  { id: 'export-formats', label: 'Export Formats' },
  { id: 'category-theory', label: 'Category Theory Export' },
  { id: 'research-vision', label: 'Research Vision' },
  { id: 'neural-architecture-search', label: 'Neural Architecture Search' },
  { id: 'catgrad-integration', label: 'Catgrad Integration' },
  { id: 'interpretability', label: 'Interpretability Research' },
  { id: 'limitations', label: 'Current Limitations' },
  { id: 'troubleshooting', label: 'Troubleshooting' },
] as const;

export const SCROLL_SPY_CONFIG = {
  OFFSET: 100,
  THRESHOLD: 0.1,
} as const;
