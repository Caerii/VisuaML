# VisuaML Library Utilities

This directory contains utility modules for the VisuaML application.

## Export and Archive Utilities

### `archiveUtils.ts`
Provides functionality for creating structured archive files containing multiple export formats.

**Key Features:**
- Creates human-readable archive format with clear file separators
- Includes metadata and extraction instructions
- Supports blob download functionality
- Generates timestamped filenames

**Main Functions:**
- `createArchiveFile(files, metadata)` - Creates a structured archive blob
- `downloadBlob(blob, filename)` - Downloads a blob as a file
- `createArchiveFilename(modelName, extension)` - Generates timestamped filenames

### `exportUtils.ts`
Handles processing and formatting of export data from different formats.

**Key Features:**
- Processes API export results into archive-ready files
- Generates user-friendly success/error messages
- Provides structured logging for export operations
- Type-safe interfaces for export data

**Main Functions:**
- `processExportResults(allResults)` - Converts API results to archive files
- `generateExportSuccessMessage(results, isArchive)` - Creates success messages
- `generateExportErrorMessage(results)` - Creates error messages
- `logExportResults(allResults, files)` - Structured console logging

## Archive Format

The archive format is a structured text file with the following layout:

```
# VisuaML Export Archive
# Model: ModelName
# Path: models.ModelName
# Generated: 2025-01-13T00:55:00.000Z
# Files: 4
# 
# File List:
#   - ModelName_hypergraph.json
#   - ModelName_hypergraph.macro
#   - ModelName_categorical_analysis.json
#   - README.md
================================================================================
FILE: ModelName_hypergraph.json
================================================================================
{
  "nodes": [...],
  "hyperedges": [...]
}
================================================================================
FILE: ModelName_hypergraph.macro
================================================================================
generate!(
  conv2d_1: Conv2d(1, 32, 3, 3),
  ...
)
================================================================================
# End of Archive
```

## Usage Example

```typescript
import { exportAllFormats } from '../api';
import { 
  createArchiveFile, 
  downloadBlob, 
  createArchiveFilename 
} from './archiveUtils';
import { 
  processExportResults, 
  generateExportSuccessMessage 
} from './exportUtils';

// Export all formats
const allResults = await exportAllFormats('models.MyModel');

if (allResults.success) {
  // Process results
  const files = processExportResults(allResults);
  
  // Create archive
  const metadata = {
    modelName: allResults.modelName,
    modelPath: allResults.modelPath,
    timestamp: allResults.timestamp,
    totalFiles: files.length
  };
  
  const archiveBlob = createArchiveFile(files, metadata);
  const filename = createArchiveFilename(allResults.modelName);
  
  // Download
  downloadBlob(archiveBlob, filename);
  
  // Show success
  const message = generateExportSuccessMessage(allResults.results, true);
  toast.success(message);
}
```

## Type Definitions

### `ArchiveFile`
```typescript
interface ArchiveFile {
  name: string;
  content: string;
  type?: string;
}
```

### `ArchiveMetadata`
```typescript
interface ArchiveMetadata {
  modelName: string;
  modelPath: string;
  timestamp: string;
  totalFiles: number;
}
```

### `AllExportsData`
```typescript
interface AllExportsData {
  modelPath: string;
  modelName: string;
  timestamp: string;
  exports: {
    json?: unknown;
    macro?: unknown;
    categorical?: unknown;
  };
  results: ExportResult[];
  success: boolean;
  readme: string;
}
``` 