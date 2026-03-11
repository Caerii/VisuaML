/**
 * Type definitions for DocsPage components
 */

export interface TOCSection {
  id: string;
  label: string;
  comingSoon?: boolean;
}

export interface DocsSectionData {
  id: string;
  title: string;
  content: string[];
  technicalDetails?: Array<{ label: string; description: string }>;
  differentiators?: Array<{ label: string; description: string }>;
  cta?: boolean;
}
