"use client";

import { useState, useEffect } from "react";
import GenerateData from "./components/GenerateData";
import SearchVector from "./components/SearchVector";
import DiscoverProcess from "./components/DiscoverProcess";
import { healthCheck } from "./lib/api";

const tabs = [
  { id: "generate" as const, label: "Generate Data" },
  { id: "search" as const, label: "Vector Search" },
  { id: "discover" as const, label: "Process Discovery" },
];

type TabId = (typeof tabs)[number]["id"];

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("generate");
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");

  useEffect(() => {
    healthCheck()
      .then(() => setBackendStatus("connected"))
      .catch(() => setBackendStatus("disconnected"));
  }, []);

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <header className="bg-white dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
              GNN Process Discovery
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-0.5">
              Automated Process Discovery using Graph Neural Networks
            </p>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                backendStatus === "connected"
                  ? "bg-emerald-500"
                  : backendStatus === "disconnected"
                    ? "bg-red-500"
                    : "bg-yellow-500 animate-pulse"
              }`}
            />
            <span className="text-zinc-500 dark:text-zinc-400">
              {backendStatus === "connected"
                ? "Backend connected"
                : backendStatus === "disconnected"
                  ? "Backend offline"
                  : "Checking..."}
            </span>
          </div>
        </div>
      </header>

      <nav className="bg-white dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? "border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400"
                    : "border-transparent text-zinc-500 hover:text-zinc-700 hover:border-zinc-300 dark:text-zinc-400 dark:hover:text-zinc-200"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === "generate" && <GenerateData />}
        {activeTab === "search" && <SearchVector />}
        {activeTab === "discover" && <DiscoverProcess />}
      </main>
    </div>
  );
}
