import { useEffect, useState, type RefObject } from 'react';

export interface UseScrollSpyOptions {
  sectionIds: readonly string[];
  offset?: number;
  containerRef?: RefObject<HTMLElement>;
}

export function useScrollSpy({ sectionIds, offset = 0, containerRef }: UseScrollSpyOptions): string {
  const [activeSection, setActiveSection] = useState<string>(sectionIds[0] || '');

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + offset;
      const container = containerRef?.current || document;

      let currentSection = sectionIds[0] || '';

      for (let i = sectionIds.length - 1; i >= 0; i--) {
        const sectionId = sectionIds[i];
        const element = container.querySelector(`#${CSS.escape(sectionId)}`) as HTMLElement;

        if (element) {
          const elementTop = element.getBoundingClientRect().top + window.scrollY;
          if (scrollPosition >= elementTop - offset) {
            currentSection = sectionId;
            break;
          }
        }
      }

      setActiveSection(currentSection);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll(); // Initial check

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [sectionIds, offset, containerRef]);

  return activeSection;
}
