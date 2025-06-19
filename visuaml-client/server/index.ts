/** @fileoverview VisuaML API server built with Fastify. Provides endpoints for model import, export, and hypergraph generation. */

import Fastify from 'fastify';
import cors from '@fastify/cors'; // Updated import
import multipart from '@fastify/multipart';
import { execa, type ExecaError } from 'execa'; // Import ExecaError type
import { z, ZodError } from 'zod'; // Import ZodError type
import * as dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { finished } from 'stream/promises';

// Type imports will be added as needed

dotenv.config(); // Load .env variables into process.env

const app = Fastify({
  logger: {
    transport: {
      target: 'pino-pretty',
      options: {
        translateTime: 'HH:MM:ss Z',
        ignore: 'pid,hostname',
      },
    },
  },
});

// Register CORS plugin
app.register(cors, {
  origin: '*', // Allow all origins for now, configure appropriately for production
});

// Register the multipart plugin
app.register(multipart);

// Define default sample inputs for models
const defaultSampleInputs: Record<string, { args: string; dtypes: string; kwargs?: string }> = {
  'models.SimpleNN': { args: '((1, 10),)', dtypes: '["float32"]' },
  'models.DemoNet': { args: '((1, 10),)', dtypes: '["long"]' },
  'models.MyTinyGPT': { args: '((1, 10),)', dtypes: '["float32"]' },
  'models.TestModel': { args: '((1, 3, 32, 32),)', dtypes: '["float32"]' },
  // Add other models from TopBar.tsx if they are different
  // and ensure their forward pass signature is compatible.
};

// Define the schema for the request body
const ImportBodySchema = z.object({
  // Regex allows alphanumeric, underscore, and dot characters for module paths
  modelPath: z.string().regex(/^[a-zA-Z0-9_.]+$/, {
    message: 'Invalid modelPath format. Only alphanumeric, underscores, and dots allowed.',
  }),
  // Add export format support
  exportFormat: z.string().default('visuaml-json'),
  // Add optional sample input overrides
  sampleInputArgs: z.string().optional(),
  sampleInputDtypes: z.array(z.string()).optional(),
});

// Define the schema for the export hypergraph request body
const ExportHypergraphRequestSchema = z.object({
  modelPath: z.string().min(1, 'Model path is required'),
  format: z.enum(['json', 'macro', 'categorical']).default('json'),
  sampleInputArgs: z.string().optional(),
  sampleInputDtypes: z.array(z.string()).optional(),
});

// Add new schema for export-all endpoint
const ExportAllRequestSchema = z.object({
  modelPath: z.string().min(1, 'Model path is required'),
  sampleInputArgs: z.string().optional(),
  sampleInputDtypes: z.array(z.string()).optional(),
});

// POST /api/upload - Handles model file upload
app.post('/api/upload', async (req, reply) => {
  const data = await req.file();
  if (!data) {
    return reply.status(400).send({ message: 'No file uploaded.' });
  }

  // Ensure the user_models directory exists
  const __filenameCurrentModule = fileURLToPath(import.meta.url);
  const __dirnameCurrentModule = path.dirname(__filenameCurrentModule);
  const projectRoot = path.resolve(__dirnameCurrentModule, '..');
  const userModelsDir = path.resolve(projectRoot, 'backend', 'user_models');

  try {
    await fs.promises.mkdir(userModelsDir, { recursive: true });
  } catch (error) {
    app.log.error('Error creating user_models directory:', error);
    return reply.status(500).send({ message: 'Failed to create directory for model.' });
  }

  // Sanitize filename and define path
  const sanitizedFilename = data.filename.replace(/[^a-zA-Z0-9_.-]/g, '_');
  const filePath = path.join(userModelsDir, sanitizedFilename);

  // Stream file to disk
  try {
    await finished(data.file.pipe(fs.createWriteStream(filePath)));
  } catch (error) {
    app.log.error('Error saving uploaded file:', error);
    return reply.status(500).send({ message: 'Failed to save uploaded file.' });
  }

  // Construct the model path for Python import
  const modelName = path.basename(sanitizedFilename, '.py');
  const modelPath = `user_models.${modelName}`;
  app.log.info(`File uploaded and saved. Processing with modelPath: ${modelPath}`);

  // Now, execute the Python script (logic adapted from /api/import)
  try {
    const scriptArgs = [modelPath];
    const scriptPath = path.resolve(projectRoot, 'backend', 'scripts', 'fx_export.py');

    app.log.info(`Executing script: python ${scriptPath} ${scriptArgs.join(' ')}`);

    const { stdout, stderr } = await execa('python', [scriptPath, ...scriptArgs], {
      cwd: projectRoot,
    });

    if (stderr) {
      // Log stderr but don't fail, as some warnings are not critical errors.
      // The Python script should exit with a non-zero code for actual errors.
      app.log.warn(`Stderr from fx_export.py: ${stderr}`);
    }

    reply.header('Content-Type', 'application/json').send(stdout);
  } catch (error) {
    app.log.error('Error executing Python script for uploaded model:', error);
    // Cleanup the uploaded file on error
    await fs.promises.unlink(filePath).catch(cleanupError => {
      app.log.error('Failed to cleanup uploaded file after error:', cleanupError);
    });

    const execaError = error as ExecaError;
    reply.status(500).send({
      message: 'Error executing Python script for uploaded model.',
      errorDetails: {
        command: execaError.command,
        exitCode: execaError.exitCode,
        stderr: execaError.stderr,
        stdout: execaError.stdout,
      },
    });
  }
});

app.post('/api/import', async (req, reply) => {
  try {
    // Validate request body
    const validationResult = ImportBodySchema.safeParse(req.body);
    if (!validationResult.success) {
      app.log.error('Invalid request body:', validationResult.error.flatten());
      return reply
        .status(400)
        .send({ message: 'Invalid request body', errors: validationResult.error.flatten() });
    }

    const { modelPath, exportFormat, sampleInputArgs, sampleInputDtypes } = validationResult.data;
    app.log.info(
      `Received import request for modelPath: ${modelPath}, exportFormat: ${exportFormat}`,
    );

    const scriptArgs = [modelPath]; // Start with modelPath

    // Add export format parameter
    if (exportFormat && exportFormat !== 'visuaml-json') {
      scriptArgs.push('--export-format', exportFormat);
    }

    // Use provided sample inputs or fall back to defaults
    let inputArgs = sampleInputArgs;
    let inputDtypes = sampleInputDtypes;

    if (!inputArgs) {
      const defaults = defaultSampleInputs[modelPath];
      if (defaults) {
        inputArgs = defaults.args;
        inputDtypes = defaults.dtypes ? JSON.parse(defaults.dtypes) : undefined;
        app.log.info(
          `Using default sample inputs for ${modelPath}: args=${inputArgs}, dtypes=${JSON.stringify(inputDtypes)}`,
        );
      } else {
        app.log.warn(
          `No default sample inputs found for ${modelPath}. Shape propagation will be skipped.`,
        );
      }
    }

    if (inputArgs) {
      scriptArgs.push('--sample-input-args', inputArgs);
    }
    if (inputDtypes && inputDtypes.length > 0) {
      scriptArgs.push('--sample-input-dtypes', JSON.stringify(inputDtypes));
    }

    // Add other default CLI args if any, e.g. --abstraction-level
    // For now, let fx_export.py use its own defaults for these.

    // Get the directory of the current module (server/index.ts)
    const __filenameCurrentModule = fileURLToPath(import.meta.url);
    const __dirnameCurrentModule = path.dirname(__filenameCurrentModule); // This will be /path/to/visuaml-client/server

    // Construct absolute path to projectRoot (visuaml-client directory)
    const projectRoot = path.resolve(__dirnameCurrentModule, '..');

    // Construct absolute path to the Python script
    const scriptPath = path.resolve(projectRoot, 'backend', 'scripts', 'fx_export.py');

    app.log.info(
      `Executing script: python ${scriptPath} ${scriptArgs.join(' ')} with CWD: ${projectRoot}`,
    );

    const { stdout, stderr } = await execa(
      'python', // Reverted from 'python3' to 'python'
      [scriptPath, ...scriptArgs], // Pass all arguments
      { cwd: projectRoot },
    );

    if (stderr) {
      app.log.error(`Error output from fx_export.py: ${stderr}`);
      // Potentially, you might want to only throw/error if execa itself errored or exit code was non-zero
    }

    app.log.info(`Script stdout length: ${stdout.length}`);
    reply.header('Content-Type', 'application/json').send(stdout);
  } catch (error: unknown) {
    // Changed from any to unknown
    app.log.error('Error in /api/import:', error);
    if (error instanceof ZodError) {
      // Check if it's a ZodError
      reply.status(400).send({ message: 'Validation error', errors: error.flatten() });
    } else if (
      error &&
      typeof error === 'object' &&
      'command' in error &&
      'failed' in error &&
      'shortMessage' in error
    ) {
      // More robust check for ExecaError like object
      // We assume it's an ExecaError based on properties. `instanceof ExecaError` is not directly possible
      // if execa is bundled or its error classes are not directly exported/re-exported in a way TS can track.
      const execaError = error as ExecaError; // Type assertion
      reply.status(500).send({
        message: 'Error executing Python script',
        errorDetails: {
          command: execaError.command,
          exitCode: execaError.exitCode,
          signal: execaError.signal,
          stderr: execaError.stderr,
          stdout: execaError.stdout,
          shortMessage: execaError.shortMessage,
          // originalMessage: execaError.originalMessage // if available
          // failed: execaError.failed,
          // timedOut: execaError.timedOut,
          // isCanceled: execaError.isCanceled,
          // killed: execaError.killed
        },
      });
    } else if (error instanceof Error) {
      // Standard Error
      reply.status(500).send({ message: 'Internal Server Error', error: error.message });
    } else {
      // Fallback for other unknown errors
      reply.status(500).send({ message: 'An unexpected error occurred' });
    }
  }
});

app.post('/api/export-hypergraph', async (req, reply) => {
  try {
    // Validate request body
    const validationResult = ExportHypergraphRequestSchema.safeParse(req.body);
    if (!validationResult.success) {
      app.log.error('Invalid request body:', validationResult.error.flatten());
      return reply
        .status(400)
        .send({ message: 'Invalid request body', errors: validationResult.error.flatten() });
    }

    const { modelPath, format, sampleInputArgs, sampleInputDtypes } = validationResult.data;
    app.log.info(
      `Received export-hypergraph request for modelPath: ${modelPath}, format: ${format}`,
    );

    // Use the same fx_export.py script with the openhg format
    const exportFormat = `openhg-${format}`;
    const scriptArgs = [modelPath, '--export-format', exportFormat];

    // Add sample inputs if provided
    if (sampleInputArgs) {
      scriptArgs.push('--sample-input-args', sampleInputArgs);
    }
    if (sampleInputDtypes && sampleInputDtypes.length > 0) {
      scriptArgs.push('--sample-input-dtypes', JSON.stringify(sampleInputDtypes));
    }

    // Get paths
    const __filenameCurrentModule = fileURLToPath(import.meta.url);
    const __dirnameCurrentModule = path.dirname(__filenameCurrentModule);
    const projectRoot = path.resolve(__dirnameCurrentModule, '..');
    const scriptPath = path.resolve(projectRoot, 'backend', 'scripts', 'fx_export.py');

    app.log.info(`Executing hypergraph export: python ${scriptPath} ${scriptArgs.join(' ')}`);

    const { stdout, stderr } = await execa('python', [scriptPath, ...scriptArgs], {
      cwd: projectRoot,
    });

    if (stderr) {
      app.log.error(`Error output from fx_export.py: ${stderr}`);
    }

    app.log.info(`Hypergraph export stdout length: ${stdout.length}`);
    reply.header('Content-Type', 'application/json').send(stdout);
  } catch (error: unknown) {
    app.log.error('Error in /api/export-hypergraph:', error);
    if (error instanceof ZodError) {
      reply.status(400).send({ message: 'Validation error', errors: error.flatten() });
    } else if (
      error &&
      typeof error === 'object' &&
      'command' in error &&
      'failed' in error &&
      'shortMessage' in error
    ) {
      const execaError = error as ExecaError;
      reply.status(500).send({
        message: 'Error executing Python export script',
        errorDetails: {
          command: execaError.command,
          exitCode: execaError.exitCode,
          signal: execaError.signal,
          stderr: execaError.stderr,
          stdout: execaError.stdout,
          shortMessage: execaError.shortMessage,
        },
      });
    } else if (error instanceof Error) {
      reply.status(500).send({ message: 'Internal Server Error', error: error.message });
    } else {
      reply.status(500).send({ message: 'An unexpected error occurred' });
    }
  }
});

app.post('/api/export-all', async (req, reply) => {
  try {
    // Validate request body
    const validationResult = ExportAllRequestSchema.safeParse(req.body);
    if (!validationResult.success) {
      app.log.error('Invalid request body:', validationResult.error.flatten());
      return reply
        .status(400)
        .send({ message: 'Invalid request body', errors: validationResult.error.flatten() });
    }

    const { modelPath, sampleInputArgs, sampleInputDtypes } = validationResult.data;
    app.log.info(`Received export-all request for modelPath: ${modelPath}`);

    // Get paths
    const __filenameCurrentModule = fileURLToPath(import.meta.url);
    const __dirnameCurrentModule = path.dirname(__filenameCurrentModule);
    const projectRoot = path.resolve(__dirnameCurrentModule, '..');
    const scriptPath = path.resolve(projectRoot, 'backend', 'scripts', 'fx_export.py');

    const modelName = modelPath.split('.').pop() || 'model';

    // Export formats to generate
    const formats = [
      { format: 'openhg-json', name: 'json' },
      { format: 'openhg-macro', name: 'macro' },
      { format: 'openhg-categorical', name: 'categorical' },
    ];

    // Generate all exports
    const exports: Record<string, unknown> = {};
    const exportResults: Array<{ format: string; name: string; success: boolean; error?: string }> =
      [];

    for (const { format, name } of formats) {
      try {
        const scriptArgs = [modelPath, '--export-format', format];

        if (sampleInputArgs) {
          scriptArgs.push('--sample-input-args', sampleInputArgs);
        }
        if (sampleInputDtypes && sampleInputDtypes.length > 0) {
          scriptArgs.push('--sample-input-dtypes', JSON.stringify(sampleInputDtypes));
        }

        app.log.info(`Generating ${format} export...`);
        const { stdout, stderr } = await execa('python', [scriptPath, ...scriptArgs], {
          cwd: projectRoot,
        });

        if (stderr) {
          app.log.warn(`Warning in ${format} export: ${stderr}`);
        }

        // Parse and store export result
        const result = JSON.parse(stdout);
        exports[name] = result;
        exportResults.push({ format, name, success: true });
        app.log.info(`✅ ${format} export completed`);
      } catch (error: unknown) {
        app.log.error(`❌ Failed to generate ${format} export:`, error);
        const errorMessage = error instanceof Error ? error.message : String(error);
        exportResults.push({ format, name, success: false, error: errorMessage });
        exports[name] = { error: errorMessage };
      }
    }

    // Create response with all exports and metadata
    const response = {
      modelPath,
      modelName,
      timestamp: new Date().toISOString(),
      exports,
      results: exportResults,
      success: exportResults.every((r) => r.success),
      readme: `# ${modelName} Export Package

Generated on: ${new Date().toISOString()}
Model: ${modelPath}

## Files Included:

1. **JSON Export** - Standard hypergraph format
   - Contains nodes, hyperedges, and metadata
   - Compatible with visualization tools

2. **Macro Export** - Rust macro syntax
   - Compatible with hellas-ai/open-hypergraphs crate
   - Ready to use in Rust projects

3. **Categorical Export** - Mathematical analysis
   - Structure analysis and complexity metrics
   - Arity information and conversion status

## Export Results:
${exportResults.map((r) => `- ${r.format}: ${r.success ? '✅ Success' : '❌ Failed - ' + r.error}`).join('\n')}

Generated by VisuaML Export System`,
    };

    reply.header('Content-Type', 'application/json').send(response);
  } catch (error: unknown) {
    app.log.error('Error in /api/export-all:', error);
    if (error instanceof ZodError) {
      reply.status(400).send({ message: 'Validation error', errors: error.flatten() });
    } else if (error instanceof Error) {
      reply.status(500).send({ message: 'Internal Server Error', error: error.message });
    } else {
      reply.status(500).send({ message: 'An unexpected error occurred' });
    }
  }
});

const PORT = process.env.PORT || 8787;
app.listen({ port: Number(PORT) }, (err) => {
  if (err) {
    app.log.error(err);
    process.exit(1);
  }
  // Logger already prints listening info, this is redundant: console.log(`API listening on :${PORT}`);
});
