"use client";

import { useState } from "react";
import { searchVector, type SearchVectorResult } from "../lib/api";

export default function SearchVector() {
  const [activity, setActivity] = useState("");
  const [graphId, setGraphId] = useState("");
  const [minCost, setMinCost] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchVectorResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await searchVector({
        activity: activity || undefined,
        graph_id: graphId || undefined,
        min_cost: minCost ? Number(minCost) : undefined,
        top_k: topK,
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
          Vector Similarity Search
        </h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
          Query the vector database for similar tasks and activities.
        </p>
      </div>

      <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Activity Name
            </label>
            <input
              type="text"
              value={activity}
              onChange={(e) => setActivity(e.target.value)}
              placeholder="e.g. Nhan_Task"
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Graph ID
            </label>
            <input
              type="text"
              value={graphId}
              onChange={(e) => setGraphId(e.target.value)}
              placeholder="e.g. G1"
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Minimum Cost
            </label>
            <input
              type="number"
              value={minCost}
              onChange={(e) => setMinCost(e.target.value)}
              placeholder="e.g. 100"
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Top K Results
            </label>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
              min={1}
              max={50}
            />
          </div>
        </div>

        <button
          onClick={handleSearch}
          disabled={loading}
          className="mt-4 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
        </div>
      )}

      {result && (
        <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-6">
          <h3 className="text-lg font-medium text-zinc-900 dark:text-zinc-50 mb-4">
            Results ({result.results.length})
          </h3>
          {result.results.length === 0 ? (
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              No results found. Try generating data first.
            </p>
          ) : (
            <div className="space-y-3">
              {result.results.map((r, i) => (
                <div
                  key={i}
                  className="border border-zinc-200 dark:border-zinc-700 rounded-lg p-4 hover:bg-zinc-50 dark:hover:bg-zinc-800/50 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-zinc-900 dark:text-zinc-100">
                      {String(r.payload.activity || "N/A")}
                    </span>
                    <span className="text-sm font-mono bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 px-2 py-0.5 rounded">
                      Score: {r.score.toFixed(4)}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm text-zinc-600 dark:text-zinc-400">
                    {Object.entries(r.payload)
                      .filter(([key]) => key !== "activity")
                      .map(([key, value]) => (
                        <div key={key}>
                          <span className="text-zinc-400 dark:text-zinc-500">
                            {key}:{" "}
                          </span>
                          <span>
                            {typeof value === "number"
                              ? value.toFixed(2)
                              : String(value)}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
