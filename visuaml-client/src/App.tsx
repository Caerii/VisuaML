import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { LandingPage } from './pages/LandingPage';
import { MainApp } from './MainApp';
import { DocsPage } from './ui/DocsPage';
import { CompilerLabPage } from './ui/CompilerLab/CompilerLabPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<MainApp />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/compiler" element={<CompilerLabPage />} />
      </Routes>
    </Router>
  );
}

export default App;
