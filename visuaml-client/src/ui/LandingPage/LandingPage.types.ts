/**
 * Type definitions for LandingPage components
 */

export interface LandingSectionData {
  id: string;
  title: string;
  content: string[];
  technicalDetails?: Array<{ label: string; description: string }>;
  differentiators?: Array<{ label: string; description: string }>;
  cta?: boolean;
}

export interface DetailItem {
  label: string;
  description: string;
}

export interface SectionButtonProps {
  children: React.ReactNode;
  color?: string;
  borderColor?: string;
  hoverBorderColor?: string;
  hoverColor?: string;
  onClick?: () => void;
}

export interface SectionDetailListProps {
  title: string;
  items: DetailItem[];
  labelColor?: string;
  descriptionColor?: string;
  introText?: string;
}
