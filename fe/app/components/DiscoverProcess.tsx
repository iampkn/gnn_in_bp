"use client";

import { useState, useRef } from "react";
import { discoverProcess, type DiscoverProcessResult } from "../lib/api";

export default function DiscoverProcess() {
  const [graphId, setGraphId] = useState("G1");
  const [beamWidth, setBeamWidth] = useState(10);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiscoverProcessResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleDiscover = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await discoverProcess({
        graph_id: file ? undefined : graphId || undefined,
        file: file || undefined,
        beam_width: beamWidth,
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
          Process Discovery
        </h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
          Run GNN inference to discover Petri net structure from event logs.
        </p>
      </div>

      <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Graph ID
            </label>
            <input
              type="text"
              value={graphId}
              onChange={(e) => setGraphId(e.target.value)}
              placeholder="e.g. G1"
              disabled={!!file}
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none disabled:bg-zinc-100 dark:disabled:bg-zinc-800 disabled:text-zinc-400"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Beam Width
            </label>
            <input
              type="number"
              value={beamWidth}
              onChange={(e) => setBeamWidth(Number(e.target.value))}
              className="w-full rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
              min={1}
              max={100}
            />
          </div>
        </div>

        <div className="mt-4">
          <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
            Or Upload Event Log CSV
          </label>
          <div className="flex items-center gap-3">
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="text-sm text-zinc-600 dark:text-zinc-400 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-100 dark:file:bg-zinc-700 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-zinc-700 dark:file:text-zinc-200 hover:file:bg-zinc-200 dark:hover:file:bg-zinc-600 file:cursor-pointer"
            />
            {file && (
              <button
                onClick={() => {
                  setFile(null);
                  if (fileRef.current) fileRef.current.value = "";
                }}
                className="text-sm text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        <button
          onClick={handleDiscover}
          disabled={loading || (!graphId && !file)}
          className="mt-4 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Discovering..." : "Discover Process"}
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
              Input Summary
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                {
                  label: "Graph ID",
                  value: result.input_info.graph_id || "Uploaded",
                },
                { label: "Traces", value: result.input_info.num_traces },
                {
                  label: "Activities",
                  value: result.input_info.num_activities,
                },
                {
                  label: "Candidate Places",
                  value: result.input_info.num_candidate_places,
                },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="bg-zinc-50 dark:bg-zinc-800 rounded-lg p-3"
                >
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {stat.label}
                  </p>
                  <p className="text-lg font-bold text-zinc-900 dark:text-zinc-50">
                    {stat.value}
                  </p>
                </div>
              ))}
            </div>

            {result.input_info.activities.length > 0 && (
              <div className="mt-4">
                <p className="text-sm font-medium text-zinc-600 dark:text-zinc-400 mb-2">
                  Activities:
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {result.input_info.activities.map((a) => (
                    <span
                      key={a}
                      className="bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 px-2 py-0.5 rounded text-xs font-mono"
                    >
                      {a}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {result.discovered_nets.map((net) => (
            <div
              key={net.rank}
              className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-zinc-900 dark:text-zinc-50">
                  Petri Net #{net.rank}
                </h3>
                <span className="text-sm font-mono bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 px-2 py-0.5 rounded">
                  log(p) = {net.log_probability.toFixed(4)}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <div className="bg-zinc-50 dark:bg-zinc-800 rounded-lg p-3">
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    Transitions
                  </p>
                  <p className="text-sm font-mono text-zinc-900 dark:text-zinc-100 mt-1">
                    {net.transitions.join(", ")}
                  </p>
                </div>
                <div className="bg-zinc-50 dark:bg-zinc-800 rounded-lg p-3">
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    Places Selected
                  </p>
                  <p className="text-lg font-bold text-zinc-900 dark:text-zinc-50">
                    {net.num_places}
                  </p>
                </div>
              </div>

              {net.places.length > 0 && (
                <div>
                  <p className="text-sm font-medium text-zinc-600 dark:text-zinc-400 mb-2">
                    Place Structure:
                  </p>
                  <div className="space-y-1.5">
                    {net.places.map((place, pi) => (
                      <div
                        key={pi}
                        className="flex items-center gap-2 text-sm font-mono bg-zinc-50 dark:bg-zinc-800 rounded px-3 py-1.5"
                      >
                        <span className="text-indigo-600 dark:text-indigo-400">
                          &#123;{place.inputs.join(", ")}&#125;
                        </span>
                        <span className="text-zinc-400">&rarr;</span>
                        <span className="text-emerald-600 dark:text-emerald-400">
                          &#123;{place.outputs.join(", ")}&#125;
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
