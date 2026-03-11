/**
 * Reusable components for DocsContent
 */
import { Stack, Typography, Box, Chip } from '@mui/material';
import { DOCS_STYLES } from './DocsContent.styles';

export const CodeBlock = ({ children }: { children: string }) => (
  <Box component="pre" sx={DOCS_STYLES.codeBlock}>
    {children}
  </Box>
);

export const StatusBadge = ({ status }: { status: 'available' | 'research' }) => (
  <Chip
    label={status === 'available' ? 'Available' : 'Research'}
    size="small"
    sx={{
      height: '20px',
      fontSize: '0.6875rem',
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.08em',
      fontFamily: "'Inter', sans-serif",
      ...(status === 'available'
        ? {
            bgcolor: 'rgba(66, 153, 225, 0.2)',
            color: 'rgba(66, 153, 225, 1)',
          }
        : {
            bgcolor: 'rgba(159, 122, 234, 0.2)',
            color: 'rgba(159, 122, 234, 1)',
          }),
      '& .MuiChip-label': {
        px: 1.25,
      },
    }}
  />
);

export const StatusHeader = ({ 
  status, 
  text 
}: { 
  status: 'available' | 'research'; 
  text: string;
}) => (
  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 0.25 }}>
    <StatusBadge status={status} />
    <Typography sx={{ ...DOCS_STYLES.body, fontSize: '0.8125rem', color: 'rgba(255, 255, 255, 0.6)' }}>
      {text}
    </Typography>
  </Box>
);

export const FeatureCard = ({ 
  title, 
  desc, 
  badge 
}: { 
  title: string; 
  desc: string; 
  badge?: 'available' | 'research';
}) => (
  <Box sx={DOCS_STYLES.featureCard}>
    <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: badge ? 0.25 : 0 }}>
      {badge && <StatusBadge status={badge} />}
      <Typography sx={DOCS_STYLES.subheading}>{title}</Typography>
    </Stack>
    <Typography sx={{ ...DOCS_STYLES.body, fontSize: '0.9375rem', mt: 0.25 }}>
      {desc}
    </Typography>
  </Box>
);

export const StatusList = ({ 
  items 
}: { 
  items: Array<{ status: string; item: string }>;
}) => (
  <Stack spacing={0.5}>
    {items.map((item, idx) => (
      <Box key={idx} sx={{ display: 'flex', gap: 1.25, alignItems: 'center' }}>
        <Typography 
          sx={{ 
            fontSize: '0.75rem', 
            fontWeight: 600,
            minWidth: item.status === 'Done' || item.status === 'Supported' ? '50px' : '75px',
            color: item.status === 'Done' || item.status === 'Supported' 
              ? 'rgba(66, 153, 225, 1)' 
              : item.status === 'Not Supported'
              ? 'rgba(255, 255, 255, 0.5)'
              : 'rgba(159, 122, 234, 1)',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
        >
          {item.status}
        </Typography>
        <Typography sx={{ ...DOCS_STYLES.body, fontSize: '0.9375rem' }}>
          {item.item}
        </Typography>
      </Box>
    ))}
  </Stack>
);

export const KeyValueList = ({ 
  items 
}: { 
  items: Array<{ name: string; desc: string }>;
}) => (
  <Stack spacing={0.5}>
    {items.map((item, idx) => (
      <Box key={idx} sx={{ display: 'flex', gap: 1.25, alignItems: 'baseline' }}>
        <Typography sx={{ ...DOCS_STYLES.subheading, minWidth: '110px', mb: 0, fontSize: '0.9375rem' }}>
          {item.name}:
        </Typography>
        <Typography sx={{ ...DOCS_STYLES.body, fontSize: '0.9375rem' }}>
          {item.desc}
        </Typography>
      </Box>
    ))}
  </Stack>
);

export const BulletList = ({ 
  items 
}: { 
  items: string[];
}) => (
  <Stack spacing={0.25}>
    {items.map((item, idx) => (
      <Typography key={idx} sx={{ ...DOCS_STYLES.body, fontSize: '0.9375rem' }}>
        • {item}
      </Typography>
    ))}
  </Stack>
);

export const ExportFormatItem = ({ 
  format, 
  desc, 
  use 
}: { 
  format: string; 
  desc: string; 
  use: string;
}) => (
  <Box>
    <Typography sx={DOCS_STYLES.heading}>{format}</Typography>
    <Typography sx={{ ...DOCS_STYLES.body, fontSize: '0.9375rem', mb: 0.125 }}>
      {desc}
    </Typography>
    <Typography sx={{ ...DOCS_STYLES.body, fontSize: '0.875rem', color: 'rgba(255, 255, 255, 0.6)' }}>
      {use}
    </Typography>
  </Box>
);

export const SectionContent = ({ 
  status, 
  statusText, 
  description, 
  children 
}: { 
  status?: 'available' | 'research';
  statusText?: string;
  description?: string;
  children: React.ReactNode;
}) => (
  <Stack spacing={1.5}>
    {status && statusText && <StatusHeader status={status} text={statusText} />}
    {description && <Typography sx={DOCS_STYLES.body}>{description}</Typography>}
    {children}
  </Stack>
);
