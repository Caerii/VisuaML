import Fastify from "fastify";
import cors from "@fastify/cors"; // Updated import
import { execa, type ExecaError } from "execa"; // Import ExecaError type
import { z, ZodError } from "zod"; // Import ZodError type
import * as dotenv from "dotenv";
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config(); // Load .env variables into process.env

const app = Fastify({ logger: true });

// Register CORS plugin
app.register(cors, {
  origin: "*", // Allow all origins for now, configure appropriately for production
});

// Define the schema for the request body
const ImportBodySchema = z.object({
  // Regex allows alphanumeric, underscore, and dot characters for module paths
  modelPath: z.string().regex(/^[a-zA-Z0-9_.]+$/,
    { message: "Invalid modelPath format. Only alphanumeric, underscores, and dots allowed." }
  )
});

app.post("/api/import", async (req, reply) => {
  try {
    // Validate request body
    const validationResult = ImportBodySchema.safeParse(req.body);
    if (!validationResult.success) {
      app.log.error("Invalid request body:", validationResult.error.flatten());
      return reply.status(400).send({ message: "Invalid request body", errors: validationResult.error.flatten() });
    }
    
    const { modelPath } = validationResult.data;
    app.log.info(`Received import request for modelPath: ${modelPath}`);

    // Get the directory of the current module (server/index.ts)
    const __filenameCurrentModule = fileURLToPath(import.meta.url);
    const __dirnameCurrentModule = path.dirname(__filenameCurrentModule); // This will be /path/to/visuaml-client/server

    // Construct absolute path to projectRoot (visuaml-client directory)
    const projectRoot = path.resolve(__dirnameCurrentModule, '..'); 

    // Construct absolute path to the Python script
    const scriptPath = path.resolve(projectRoot, 'backend', 'scripts', 'fx_export.py');

    app.log.info(`Executing script: python ${scriptPath} ${modelPath} with CWD: ${projectRoot}`);

    const { stdout, stderr } = await execa("python", [
      scriptPath, // Use absolute path to script
      modelPath,
    ], { cwd: projectRoot }); // Keep cwd as projectRoot for the script's internal relative path access (e.g. to models/)

    if (stderr) {
      app.log.error(`Error output from fx_export.py: ${stderr}`);
      // Potentially, you might want to only throw/error if execa itself errored or exit code was non-zero
    }
    
    app.log.info(`Script stdout length: ${stdout.length}`);
    reply.header("Content-Type", "application/json").send(stdout);

  } catch (error: unknown) { // Changed from any to unknown
    app.log.error("Error in /api/import:", error);
    if (error instanceof ZodError) { // Check if it's a ZodError
        reply.status(400).send({ message: "Validation error", errors: error.flatten() });
    } else if (error && typeof error === 'object' && 'command' in error && 'failed' in error && 'shortMessage' in error) { // More robust check for ExecaError like object
        // We assume it's an ExecaError based on properties. `instanceof ExecaError` is not directly possible
        // if execa is bundled or its error classes are not directly exported/re-exported in a way TS can track.
        const execaError = error as ExecaError; // Type assertion
        reply.status(500).send({ 
            message: "Error executing Python script", 
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
            }
        });
    } else if (error instanceof Error) { // Standard Error
        reply.status(500).send({ message: "Internal Server Error", error: error.message });
    } else { // Fallback for other unknown errors
        reply.status(500).send({ message: "An unexpected error occurred" });
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