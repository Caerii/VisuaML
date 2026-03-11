/**
 * Documentation content sections
 * Unified, compressed, and organized design
 */
import { Stack, Typography, Box } from '@mui/material';
import { DocsSection } from './DocsSection';
import { DOCS_STYLES } from './DocsContent.styles';
import {
  CodeBlock,
  FeatureCard,
  StatusList,
  KeyValueList,
  BulletList,
  ExportFormatItem,
  SectionContent,
} from './DocsContent.components';
import { DOCS_DATA, CODE_SNIPPETS } from './DocsContent.data';

export function DocsContent() {
  return (
    <>
      <DocsSection id="get-started" title="Get Started">
        <Stack spacing={1.5}>
          <Typography sx={DOCS_STYLES.body}>
            Upload a PyTorch model file and explore its architecture. VisuaML uses PyTorch FX tracing to extract computational graphs.
          </Typography>
          <Box component="ol" sx={{ pl: 3, m: 0 }}>
            {DOCS_DATA.getStarted.map((step, idx) => (
              <Typography key={idx} component="li" sx={{ ...DOCS_STYLES.listItem, mb: 0.75 }}>
                {step}
              </Typography>
            ))}
          </Box>
        </Stack>
      </DocsSection>

      <DocsSection id="features" title="Features">
        <SectionContent
          description="Real-time collaborative visualization with categorical export—bridges PyTorch and formal mathematics."
        >
          <Stack spacing={1}>
            {DOCS_DATA.features.map((f, idx) => (
              <FeatureCard key={idx} title={f.title} desc={f.desc} />
            ))}
          </Stack>
        </SectionContent>
      </DocsSection>

      <DocsSection id="usage" title="Usage">
        <SectionContent description="Load a model from the dropdown or upload a `.py` file. Explore the graph, view node details, and export to multiple formats.">
          <CodeBlock>{CODE_SNIPPETS.usage}</CodeBlock>
        </SectionContent>
      </DocsSection>

      <DocsSection id="architecture" title="Architecture">
        <SectionContent description="Full-stack application: React frontend, Node.js API server, Python backend with PyTorch FX.">
          <KeyValueList items={DOCS_DATA.architecture as unknown as { name: string; desc: string }[]} />
          <CodeBlock>{CODE_SNIPPETS.architecture}</CodeBlock>
        </SectionContent>
      </DocsSection>

      <DocsSection id="categorical-foundation" title="Categorical Foundation">
        <SectionContent
          status="available"
          statusText="Fully implemented"
          description="Layers become typed morphisms with automatic composition validation. Enables formal reasoning about model structure."
        >
          <CodeBlock>{CODE_SNIPPETS.categorical}</CodeBlock>
          <KeyValueList items={DOCS_DATA.categoricalFoundation as unknown as { name: string; desc: string }[]} />
        </SectionContent>
      </DocsSection>

      <DocsSection id="bridge-architecture" title="Bridge Architecture">
        <SectionContent
          status="available"
          statusText="Complete pipeline tested"
          description="Works with existing PyTorch models—no rewriting required. Applies categorical mathematics while preserving your workflow."
        >
          <CodeBlock>{CODE_SNIPPETS.bridge}</CodeBlock>
          <BulletList items={DOCS_DATA.bridgeBenefits as unknown as string[]} />
        </SectionContent>
      </DocsSection>

      <DocsSection id="collaboration" title="Real-time Collaboration">
        <SectionContent description="Multiple users explore the same model simultaneously with live cursors and synchronized state.">
          <Stack spacing={1}>
            {DOCS_DATA.collaboration.map((item, idx) => (
              <FeatureCard key={idx} title={item.title} desc={item.desc} />
            ))}
          </Stack>
        </SectionContent>
      </DocsSection>

      <DocsSection id="export-formats" title="Export Formats">
        <SectionContent description="Export to multiple formats, generated from the categorical representation.">
          <Stack spacing={1.25}>
            {DOCS_DATA.exportFormats.map((item, idx) => (
              <ExportFormatItem key={idx} format={item.format} desc={item.desc} use={item.use} />
            ))}
          </Stack>
        </SectionContent>
      </DocsSection>

      <DocsSection id="category-theory" title="Category Theory Export">
        <SectionContent
          status="available"
          statusText="Unique capability"
          description="Open hypergraphs provide superior representation with boundary structure, compositional interface, and type-aware connections."
        >
          <Stack spacing={1}>
            {DOCS_DATA.categoryTheory.map((item, idx) => (
              <FeatureCard key={idx} title={item.title} desc={item.desc} />
            ))}
          </Stack>
        </SectionContent>
      </DocsSection>

      <DocsSection id="research-vision" title="Research Vision">
        <SectionContent
          status="research"
          statusText="Active research directions"
          description="Our categorical foundation enables research directions that could transform neural network understanding and design."
        >
          <Stack spacing={1}>
            {DOCS_DATA.researchVision.map((item, idx) => (
              <FeatureCard key={idx} title={item.title} desc={item.desc} badge="research" />
            ))}
          </Stack>
        </SectionContent>
      </DocsSection>

      <DocsSection id="neural-architecture-search" title="Neural Architecture Search">
        <SectionContent
          status="research"
          statusText="Research vision"
          description="Transform architecture search from discrete optimization over fixed templates into type-safe compositional exploration over the entire ML ecosystem."
        >
          <StatusList items={DOCS_DATA.nasStatus as unknown as Array<{ status: string; item: string }>} />
          <Typography sx={{ ...DOCS_STYLES.body, fontSize: '0.875rem', color: 'rgba(255, 255, 255, 0.6)' }}>
            See{' '}
            <Box component="span" sx={{ color: 'rgba(66, 153, 225, 1)' }}>
              Categorical Neural Architecture Search
            </Box>
            {' '}research document for detailed analysis.
          </Typography>
        </SectionContent>
      </DocsSection>

      <DocsSection id="catgrad-integration" title="Catgrad Integration">
        <SectionContent
          status="research"
          statusText="Research direction"
          description="Catgrad is a categorical deep learning compiler that could accelerate architecture evaluation via framework-free static compilation."
        >
          <Stack spacing={1}>
            {DOCS_DATA.catgradFeatures.map((item, idx) => (
              <FeatureCard key={idx} title={item.title} desc={item.desc} />
            ))}
          </Stack>
          <StatusList items={DOCS_DATA.catgradStatus as unknown as Array<{ status: string; item: string }>} />
        </SectionContent>
      </DocsSection>

      <DocsSection id="interpretability" title="Interpretability Research">
        <SectionContent
          status="research"
          statusText="Research vision"
          description="The Anthropic study required 25+ researchers for months to understand 30% of a model. Categorical structure could enable distributed, collaborative interpretability."
        >
          <Stack spacing={1}>
            {DOCS_DATA.interpretability.map((item, idx) => (
              <FeatureCard key={idx} title={item.title} desc={item.desc} />
            ))}
          </Stack>
          <StatusList items={DOCS_DATA.interpretabilityStatus as unknown as Array<{ status: string; item: string }>} />
        </SectionContent>
      </DocsSection>

      <DocsSection id="limitations" title="Limitations">
        <SectionContent description="Current constraints and unimplemented features.">
          <Stack spacing={1.25}>
            <Box>
              <Typography sx={DOCS_STYLES.heading}>Framework Support</Typography>
              <Box sx={{ mt: 0.5 }}>
                <StatusList items={DOCS_DATA.limitations.framework as unknown as Array<{ status: string; item: string }>} />
              </Box>
            </Box>
            <Box>
              <Typography sx={DOCS_STYLES.heading}>Model Compatibility</Typography>
              <Stack spacing={0.75} sx={{ mt: 0.5 }}>
                {DOCS_DATA.limitations.compatibility.map((item, idx) => (
                  <Box key={idx}>
                    <Typography sx={{ ...DOCS_STYLES.subheading, fontSize: '0.9375rem', mb: 0.125 }}>
                      {item.issue}
                    </Typography>
                    <Typography sx={{ ...DOCS_STYLES.body, fontSize: '0.9375rem' }}>
                      {item.solution}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </Box>
          </Stack>
        </SectionContent>
      </DocsSection>

      <DocsSection id="troubleshooting" title="Troubleshooting">
        <Stack spacing={1}>
          {DOCS_DATA.troubleshooting.map((item, idx) => (
            <FeatureCard key={idx} title={item.problem} desc={item.solution} />
          ))}
        </Stack>
      </DocsSection>
    </>
  );
}
