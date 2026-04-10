"use client";

import { useState } from "react";
import { generateData, type GenerateDataResult } from "../lib/api";

export default function GenerateData() {
  const [numVariants, setNumVariants] = useState(20);
  const [tracesPerGraph, setTracesPerGraph] = useState(200);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateDataResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await generateData({
        num_variants: numVariants,
        traces_per_graph: tracesPerGraph,
        seed,
      });
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50">
          Generate Event Log Data
        </h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
          Run Phase 1 pipeline to generate graph variants and simulate event
          logs.
        </p>
      </div>

      <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Number of Variants
            </label>
            <input
              type="number"
              value={numVariants}
              onChange={(e) => setNumVariants(Number(e.target.value))}
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
              min={1}
              max={100}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Traces per Graph
            </label>
            <input
              type="number"
              value={tracesPerGraph}
              onChange={(e) => setTracesPerGraph(Number(e.target.value))}
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
              min={10}
              max={1000}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Random Seed
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
            />
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={loading}
          className="mt-4 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Generating..." : "Generate Data"}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-6">
            <h3 className="text-lg font-medium text-zinc-900 dark:text-zinc-50 mb-4">
              Summary
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { label: "Total Graphs", value: result.total_graphs },
                {
                  label: "Total Events",
                  value: result.total_events.toLocaleString(),
                },
                {
                  label: "Total Cases",
                  value: result.total_cases.toLocaleString(),
                },
                { label: "Unique Activities", value: result.unique_activities },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="bg-zinc-50 dark:bg-zinc-800 rounded-lg p-3"
                >
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {stat.label}
                  </p>
                  <p className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
                    {stat.value}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {result.per_graph && result.per_graph.length > 0 && (
            <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-6">
              <h3 className="text-lg font-medium text-zinc-900 dark:text-zinc-50 mb-4">
                Per-Graph Details
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-zinc-200 dark:border-zinc-700">
                      <th className="text-left py-2 px-3 font-medium text-zinc-600 dark:text-zinc-400">
                        Graph ID
                      </th>
                      <th className="text-right py-2 px-3 font-medium text-zinc-600 dark:text-zinc-400">
                        Cases
                      </th>
                      <th className="text-right py-2 px-3 font-medium text-zinc-600 dark:text-zinc-400">
                        Events
                      </th>
                      <th className="text-right py-2 px-3 font-medium text-zinc-600 dark:text-zinc-400">
                        Total Cost
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.per_graph.map((g) => (
                      <tr
                        key={g.graph_id}
                        className="border-b border-zinc-100 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
                      >
                        <td className="py-2 px-3 font-mono text-zinc-900 dark:text-zinc-100">
                          {g.graph_id}
                        </td>
                        <td className="text-right py-2 px-3 text-zinc-700 dark:text-zinc-300">
                          {g.cases}
                        </td>
                        <td className="text-right py-2 px-3 text-zinc-700 dark:text-zinc-300">
                          {g.events.toLocaleString()}
                        </td>
                        <td className="text-right py-2 px-3 text-zinc-700 dark:text-zinc-300">
                          {g.total_cost.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
