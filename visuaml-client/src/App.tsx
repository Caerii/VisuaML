import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { LandingPage } from './pages/LandingPage';
import { MainApp } from './MainApp';
import { DocsPage } from './ui/DocsPage';
import { CompilerLabPage } from './ui/CompilerLab/CompilerLabPage';
import { ArchitectureAtlasPage } from './ui/ArchitectureAtlas/ArchitectureAtlasPage';
import { RoadmapPage } from './ui/Roadmap/RoadmapPage';
import { ResearchEvidencePage } from './ui/ResearchEvidence/ResearchEvidencePage';
import { ResearchLibraryPage } from './ui/ResearchEvidence/ResearchLibraryPage';
import { ResearchKnowledgePage } from './ui/ResearchEvidence/ResearchKnowledgePage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<MainApp />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/compiler" element={<CompilerLabPage />} />
        <Route path="/architectures" element={<ArchitectureAtlasPage />} />
        <Route path="/roadmap" element={<RoadmapPage />} />
        <Route path="/research" element={<ResearchEvidencePage />} />
        <Route path="/research/library" element={<ResearchLibraryPage />} />
        <Route path="/research/knowledge" element={<ResearchKnowledgePage />} />
      </Routes>
    </Router>
  );
}

export default App;
