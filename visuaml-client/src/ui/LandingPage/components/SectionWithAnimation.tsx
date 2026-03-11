/**
 * Animated section wrapper component
 * Provides fade-in animation when section comes into view
 */
import { useInView } from 'react-intersection-observer';
import { motion } from 'framer-motion';

interface SectionWithAnimationProps {
  children: React.ReactNode;
  delay?: number;
}

export function SectionWithAnimation({ children, delay = 0 }: SectionWithAnimationProps) {
  const { ref, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true,
  });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 50 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
      transition={{ duration: 0.6, delay }}
    >
      {children}
    </motion.div>
  );
}
