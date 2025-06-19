/** @fileoverview Defines the TopBar component, which includes controls for selecting a model and initiating the import process. It interacts with the Yjs document for shared state and the Zustand store for network facts. */
import { useTopBar } from './useTopBar';
import { AVAILABLE_MODELS } from './TopBar.model';
import { useRef } from 'react';

const TopBar = () => {
  const {
    modelPath,
    isLoadingUI,
    isUploading,
    isExporting,
    exportFormat,
    handleImportClick,
    handleFileUpload,
    handleModelChange,
    handleExport,
    handleExportFormatChange,
  } = useTopBar();

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  return (
    <header className="topbar">
      <div className="topbar__container">
        <a href="/" className="topbar__logo-link">
          <img
            src="/visuaml_logo.png"
            alt="VisuaML Logo"
            className="topbar__logo"
          />
        </a>
        
        <div className="topbar__controls">
          <div className="topbar__control-group">
            <label htmlFor="modelPathSelect" className="topbar__label">
              Models:
            </label>
            <select
              id="modelPathSelect"
              value={modelPath}
              onChange={handleModelChange}
              className="topbar__select"
              disabled={isLoadingUI || isExporting || isUploading}
            >
              <optgroup label="✅ Fixed Models (Export Compatible)">
                {AVAILABLE_MODELS.filter((m) => m.category === 'fixed').map((model) => (
                  <option key={model.value} value={model.value} title={model.description}>
                    {model.label}
                  </option>
                ))}
              </optgroup>
              <optgroup label="🟢 Working Models">
                {AVAILABLE_MODELS.filter((m) => m.category === 'working').map((model) => (
                  <option key={model.value} value={model.value} title={model.description}>
                    {model.label}
                  </option>
                ))}
              </optgroup>
              <optgroup label="❌ Original Models (For Comparison)">
                {AVAILABLE_MODELS.filter((m) => m.category === 'original').map((model) => (
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
              className="topbar__button"
              disabled={isLoadingUI || isExporting || isUploading}
            >
              {isLoadingUI ? 'Importing...' : 'Import'}
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              style={{ display: 'none' }}
              accept=".py"
            />
            <button
              onClick={handleUploadButtonClick}
              className="topbar__button--secondary"
              disabled={isLoadingUI || isExporting || isUploading}
            >
              {isUploading ? 'Uploading...' : 'Upload .py'}
            </button>
          </div>

          <div className="topbar__control-group">
            <label htmlFor="exportFormatSelect" className="topbar__label">
              Export:
            </label>
            <select
              id="exportFormatSelect"
              value={exportFormat}
              onChange={handleExportFormatChange}
              className="topbar__select"
              disabled={isLoadingUI || isExporting}
            >
              <option value="json">JSON (Frontend)</option>
              <option value="macro">Macro (Rust)</option>
              <option value="categorical">Categorical (Python)</option>
              <option value="all">📦 All Formats (Archive)</option>
            </select>
            <button
              onClick={handleExport}
              className="topbar__button--secondary"
              disabled={
                isLoadingUI ||
                isExporting ||
                !AVAILABLE_MODELS.find((m) => m.value === modelPath)?.exportCompatible
              }
              title={
                !AVAILABLE_MODELS.find((m) => m.value === modelPath)?.exportCompatible
                  ? 'This model is not compatible with open-hypergraph export'
                  : 'Export to open-hypergraph format'
              }
            >
              {isExporting 
                ? 'Exporting...' 
                : exportFormat === 'json' 
                  ? 'Export VisuaML Graph'
                  : exportFormat === 'macro'
                    ? 'Export Hypergraph as HyperSyn Macro'
                    : exportFormat === 'categorical'
                      ? 'Export as Open Hypergraph'
                      : '📦 Export All Formats'
              }
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default TopBar;
