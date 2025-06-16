/** @fileoverview Defines the TopBar component, which includes controls for selecting a model and initiating the import process. It interacts with the Yjs document for shared state and the Zustand store for network facts. */
import { Box } from '@mui/material'; // For layout
import { useTopBar } from './useTopBar';
import { AVAILABLE_MODELS } from './TopBar.model';
import styles from './styles/TopBar.module.css'; 

const TopBar = () => {
  const {
    modelPath,
    isLoadingUI,
    isExporting,
    exportFormat,
    handleImportClick,
    handleModelChange,
    handleExport,
    handleExportFormatChange,
  } = useTopBar();

  return (
    <header className={styles.header}>
      {/* Use MUI Box for flex layout of logo and controls */}
      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}> 
        <a href="/" className={styles.logoLink}>
          <img 
            src="/visuaml_logo.png" // Path relative to public folder
            alt="VisuaML Logo"
            className={styles.logo} // We will define this style in the CSS module
          />
        </a>
        {/* Group for model selection and import button */}
        <Box sx={{ display: 'flex', alignItems: 'center', ml: 1 }}>
          <label htmlFor="modelPathSelect" className={styles.label}>
            Models:
          </label>
          <select
            id="modelPathSelect"
            value={modelPath}
            onChange={handleModelChange}
            className={styles.select}
            disabled={isLoadingUI || isExporting} 
          >
            <optgroup label="✅ Fixed Models (Export Compatible)">
              {AVAILABLE_MODELS.filter(m => m.category === 'fixed').map((model) => (
                <option key={model.value} value={model.value} title={model.description}>
                  {model.label}
                </option>
              ))}
            </optgroup>
            <optgroup label="🟢 Working Models">
              {AVAILABLE_MODELS.filter(m => m.category === 'working').map((model) => (
                <option key={model.value} value={model.value} title={model.description}>
                  {model.label}
                </option>
              ))}
            </optgroup>
            <optgroup label="❌ Original Models (For Comparison)">
              {AVAILABLE_MODELS.filter(m => m.category === 'original').map((model) => (
                <option key={model.value} value={model.value} title={model.description}>
                  {model.label}
                </option>
              ))}
            </optgroup>
            <optgroup label="Actions">
              <option value="add_new" disabled>
                + Add New Model (Coming Soon)
              </option>
            </optgroup>
          </select>
          <button
            onClick={handleImportClick}
            className={styles.button}
            disabled={isLoadingUI || isExporting} 
          >
            {isLoadingUI ? 'Importing...' : 'Import'}
          </button>
          
          {/* Export controls */}
          <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
            <label htmlFor="exportFormatSelect" className={styles.label}>
              Export:
            </label>
            <select
              id="exportFormatSelect"
              value={exportFormat}
              onChange={handleExportFormatChange}
              className={styles.select}
              disabled={isLoadingUI || isExporting}
            >
              <option value="json">JSON (Frontend)</option>
              <option value="macro">Macro (Rust)</option>
              <option value="categorical">Categorical (Python)</option>
              <option value="all">📦 All Formats (Archive)</option>
            </select>
            <button
              onClick={handleExport}
              className={styles.button}
              disabled={isLoadingUI || isExporting || !AVAILABLE_MODELS.find(m => m.value === modelPath)?.exportCompatible}
              title={!AVAILABLE_MODELS.find(m => m.value === modelPath)?.exportCompatible ? 'This model is not compatible with open-hypergraph export' : 'Export to open-hypergraph format'}
            >
              {isExporting ? 'Exporting...' : 'Export HG'}
            </button>
          </Box>
        </Box>
      </Box>
    </header>
  );
};

export default TopBar; 