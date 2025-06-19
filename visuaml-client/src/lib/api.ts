/** @fileoverview This file contains API service functions for interacting with the backend. */

import type {
  ImportApiResponse,
  ExportHypergraphResponse,
  ExportFormat,
} from '../ui/TopBar/TopBar.model';

/**
 * Fetches a model from the backend and returns it.
 * @param modelPath The Python path to the model to import.
 * @param exportFormat Optional export format for immediate export during import.
 * @param sampleInputArgs Optional sample input arguments override.
 * @param sampleInputDtypes Optional sample input dtypes override.
 * @returns A promise that resolves to the imported model data.
 * @throws An error if the API response is not ok or the data format is invalid.
 */
export const importModel = async (
  modelPath: string,
  exportFormat: string = 'visuaml-json',
  sampleInputArgs?: string,
  sampleInputDtypes?: string[],
): Promise<ImportApiResponse> => {
  const response = await fetch('/api/import', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelPath,
      exportFormat,
      sampleInputArgs,
      sampleInputDtypes,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ message: response.statusText }));
    const errorMessage =
      errData.message ||
      errData.error ||
      JSON.stringify(errData) ||
      `API Error: ${response.status}`;
    throw new Error(errorMessage);
  }

  const importedData = await response.json();
  if (!importedData.nodes || !importedData.edges) {
    throw new Error('Invalid data format from API: nodes or edges missing.');
  }

  return importedData;
};

/**
 * Exports a model to open-hypergraph format using the dedicated export endpoint.
 * @param modelPath The Python path to the model to export.
 * @param format The export format ('json' or 'macro').
 * @param sampleInputArgs Optional sample input arguments for the model.
 * @param sampleInputDtypes Optional sample input dtypes for the model.
 * @returns A promise that resolves to the exported hypergraph data.
 * @throws An error if the API response is not ok or the data format is invalid.
 */
export const exportModelHypergraph = async (
  modelPath: string,
  format: ExportFormat = 'json',
  sampleInputArgs?: string,
  sampleInputDtypes?: string[],
): Promise<ExportHypergraphResponse> => {
  const response = await fetch('/api/export-hypergraph', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelPath,
      format,
      sampleInputArgs,
      sampleInputDtypes,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ message: response.statusText }));
    const errorMessage =
      errData.message ||
      errData.error ||
      JSON.stringify(errData) ||
      `API Error: ${response.status}`;
    throw new Error(errorMessage);
  }

  const exportedData = await response.json();
  return {
    ...exportedData,
    success: true,
  };
};

/**
 * Import a model with simultaneous export to open-hypergraph format.
 * This uses the enhanced import endpoint that supports export formats.
 * @param modelPath The Python path to the model to import.
 * @param exportFormat The export format ('openhg-json', 'openhg-macro', 'openhg-categorical').
 * @param sampleInputArgs Optional sample input arguments for the model.
 * @param sampleInputDtypes Optional sample input dtypes for the model.
 * @returns A promise that resolves to both graph data and export data.
 */
export const importModelWithExport = async (
  modelPath: string,
  exportFormat: 'openhg-json' | 'openhg-macro' | 'openhg-categorical',
  sampleInputArgs?: string,
  sampleInputDtypes?: string[],
): Promise<ImportApiResponse & ExportHypergraphResponse> => {
  const response = await fetch('/api/import', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelPath,
      exportFormat,
      sampleInputArgs,
      sampleInputDtypes,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ message: response.statusText }));
    const errorMessage =
      errData.message ||
      errData.error ||
      JSON.stringify(errData) ||
      `API Error: ${response.status}`;
    throw new Error(errorMessage);
  }

  const result = await response.json();

  // For openhg formats, the response includes both graph data and export data
  if (!result.nodes || !result.edges) {
    throw new Error('Invalid data format from API: nodes or edges missing.');
  }

  return {
    ...result,
    success: true,
  };
};

/**
 * Export all formats for a model (JSON, Macro, Categorical) in a single request.
 * @param modelPath The Python path to the model to export.
 * @param sampleInputArgs Optional sample input arguments for the model.
 * @param sampleInputDtypes Optional sample input dtypes for the model.
 * @returns A promise that resolves to all export formats and metadata.
 */
export const exportAllFormats = async (
  modelPath: string,
  sampleInputArgs?: string,
  sampleInputDtypes?: string[],
): Promise<import('./exportUtils').AllExportsData> => {
  const response = await fetch('/api/export-all', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      modelPath,
      sampleInputArgs,
      sampleInputDtypes,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ message: response.statusText }));
    const errorMessage =
      errData.message ||
      errData.error ||
      JSON.stringify(errData) ||
      `API Error: ${response.status}`;
    throw new Error(errorMessage);
  }

  return await response.json();
};

/**
 * Uploads a model file to the backend for processing.
 * @param file The .py model file to upload.
 * @returns A promise that resolves to the imported model data.
 * @throws An error if the API response is not ok or the data format is invalid.
 */
export const uploadModel = async (file: File): Promise<ImportApiResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
    // Note: 'Content-Type' header is not set manually for FormData.
    // The browser will set it automatically with the correct boundary.
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => ({ message: response.statusText }));
    const errorMessage =
      errData.message ||
      errData.error?.stderr || // Use stderr from execa error if available
      JSON.stringify(errData.errorDetails) ||
      JSON.stringify(errData) ||
      `API Error: ${response.status}`;
    throw new Error(errorMessage);
  }

  const importedData = await response.json();
  if (!importedData.nodes || !importedData.edges) {
    throw new Error('Invalid data format from API: nodes or edges missing.');
  }

  return importedData;
};
