import { useState } from "react";
import { useSyncedGraphActions } from "../y/useSyncedGraph"; // Corrected path
import { autoLayout } from "../lib/autoLayout.ts"; // Try with .ts extension
import type { Node, Edge } from "@xyflow/react"; // For typing response from API
import { toast } from "sonner"; // Import toast

// Define structure for API response if not already globally available
interface ImportApiResponse {
  nodes: Node[];
  edges: Edge[];
}

const TopBar = () => {
  const [modelPath, setModelPath] = useState("models.MyTinyGPT"); // Default model path
  const [isLoading, setIsLoading] = useState(false);
  const { commitNodes, commitEdges } = useSyncedGraphActions();

  const handleImportClick = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ modelPath }), // Use modelPath from state
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ message: response.statusText }));
        const errorMessage = errData.message || errData.error || JSON.stringify(errData) || `API Error: ${response.status}`;
        throw new Error(errorMessage);
      }

      const importedData = await response.json() as ImportApiResponse;
      
      if (!importedData.nodes || !importedData.edges) {
        throw new Error("Invalid data format from API: nodes or edges missing.");
      }
      
      // Assuming autoLayout is synchronous for now, as per playbook snippet
      const laidOutData = autoLayout(importedData.nodes, importedData.edges);
      
      commitNodes(laidOutData.nodes);
      commitEdges(laidOutData.edges);
      toast.success(`Model '${modelPath}' imported successfully!`);

    } catch (err: unknown) {
      console.error("Failed to import model:", err);
      const message = err instanceof Error ? err.message : "An unknown error occurred during import.";
      toast.error(message, { duration: 5000 });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <header className="flex items-center gap-2 p-2 bg-primary-700 text-white shadow-md">
      <label htmlFor="modelPathInput" className="text-sm font-medium">Model Path:</label>
      <input
        id="modelPathInput"
        type="text"
        value={modelPath}
        onChange={(e) => setModelPath(e.target.value)}
        placeholder="e.g., models.MyTinyGPT"
        className="flex-grow rounded px-2 py-1 text-black focus:ring-2 focus:ring-accent-400 focus:border-accent-400 outline-none w-72"
        disabled={isLoading}
      />
      <button
        onClick={handleImportClick}
        className="bg-accent-400 hover:bg-yellow-500 px-4 py-1 rounded-md text-sm font-semibold text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-150"
        disabled={isLoading}
      >
        {isLoading ? "Importing..." : "Import"}
      </button>
    </header>
  );
};

export default TopBar; 