/**
 * Type definitions for DocsTableOfContents component
 */
import type { TOCSection } from './DocsPage.types';

export interface DocsTableOfContentsProps {
  sections: readonly TOCSection[];
  activeSection: string;
  onSectionClick: (id: string) => void;
  mobile?: boolean;
}
