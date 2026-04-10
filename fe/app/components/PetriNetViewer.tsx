"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  getPetriNet,
  type PetriNetResult,
  type PetriNetNode,
  type PetriNetEdge,
} from "../lib/api";

const NODE_W = 150;
const NODE_H = 44;
const CIRCLE_R = 22;

const TYPE_COLORS: Record<string, { fill: string; stroke: string; text: string }> = {
  StartEvent:       { fill: "#d1fae5", stroke: "#059669", text: "#065f46" },
  EndEvent:         { fill: "#fee2e2", stroke: "#dc2626", text: "#991b1b" },
  Task:             { fill: "#e0e7ff", stroke: "#4f46e5", text: "#312e81" },
  ExclusiveGateway: { fill: "#fef9c3", stroke: "#ca8a04", text: "#854d0e" },
};

function nodeCenter(n: PetriNetNode): { cx: number; cy: number } {
  if (n.type === "StartEvent" || n.type === "EndEvent") {
    return { cx: n.x + CIRCLE_R, cy: n.y };
  }
  if (n.type === "ExclusiveGateway") {
    return { cx: n.x + 24, cy: n.y };
  }
  return { cx: n.x + NODE_W / 2, cy: n.y };
}

function nodeBorderPoint(
  n: PetriNetNode,
  tx: number,
  ty: number,
): { x: number; y: number } {
  const { cx, cy } = nodeCenter(n);
  const dx = tx - cx;
  const dy = ty - cy;
  const angle = Math.atan2(dy, dx);

  if (n.type === "StartEvent" || n.type === "EndEvent") {
    return { x: cx + CIRCLE_R * Math.cos(angle), y: cy + CIRCLE_R * Math.sin(angle) };
  }
  if (n.type === "ExclusiveGateway") {
    const r = 24;
    const absCos = Math.abs(Math.cos(angle));
    const absSin = Math.abs(Math.sin(angle));
    const dist = r / Math.max(absCos + absSin, 0.01);
    return { x: cx + dist * Math.cos(angle), y: cy + dist * Math.sin(angle) };
  }
  const hw = NODE_W / 2;
  const hh = NODE_H / 2;
  const absCos = Math.abs(Math.cos(angle));
  const absSin = Math.abs(Math.sin(angle));
  let scale: number;
  if (absCos * hh > absSin * hw) {
    scale = hw / Math.max(absCos, 0.001);
  } else {
    scale = hh / Math.max(absSin, 0.001);
  }
  return { x: cx + scale * Math.cos(angle), y: cy + scale * Math.sin(angle) };
}

function NodeShape({ node, selected, onSelect }: {
  node: PetriNetNode;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const colors = TYPE_COLORS[node.type] || TYPE_COLORS.Task;
  const outlineClass = selected ? "drop-shadow-lg" : "";

  if (node.type === "StartEvent" || node.type === "EndEvent") {
    return (
      <g
        onClick={() => onSelect(node.id)}
        className={`cursor-pointer ${outlineClass}`}
      >
        <circle
          cx={node.x + CIRCLE_R}
          cy={node.y}
          r={CIRCLE_R}
          fill={colors.fill}
          stroke={selected ? "#1d4ed8" : colors.stroke}
          strokeWidth={selected ? 3 : 2}
        />
        <text
          x={node.x + CIRCLE_R}
          y={node.y + 1}
          textAnchor="middle"
          dominantBaseline="central"
          className="text-[10px] font-bold select-none pointer-events-none"
          fill={colors.text}
        >
          {node.type === "StartEvent" ? "S" : "E"}
        </text>
        <text
          x={node.x + CIRCLE_R}
          y={node.y + CIRCLE_R + 14}
          textAnchor="middle"
          className="text-[10px] select-none pointer-events-none"
          fill="#71717a"
        >
          {node.label.replace(/_/g, " ")}
        </text>
      </g>
    );
  }

  if (node.type === "ExclusiveGateway") {
    const cx = node.x + 24;
    const cy = node.y;
    const r = 24;
    const points = `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`;
    return (
      <g
        onClick={() => onSelect(node.id)}
        className={`cursor-pointer ${outlineClass}`}
      >
        <polygon
          points={points}
          fill={colors.fill}
          stroke={selected ? "#1d4ed8" : colors.stroke}
          strokeWidth={selected ? 3 : 2}
        />
        <text
          x={cx}
          y={cy + 1}
          textAnchor="middle"
          dominantBaseline="central"
          className="text-[14px] font-bold select-none pointer-events-none"
          fill={colors.text}
        >
          X
        </text>
        <text
          x={cx}
          y={cy + r + 14}
          textAnchor="middle"
          className="text-[10px] select-none pointer-events-none"
          fill="#71717a"
        >
          {node.label.replace(/_/g, " ")}
        </text>
      </g>
    );
  }

  return (
    <g
      onClick={() => onSelect(node.id)}
      className={`cursor-pointer ${outlineClass}`}
    >
      <rect
        x={node.x}
        y={node.y - NODE_H / 2}
        width={NODE_W}
        height={NODE_H}
        rx={8}
        fill={colors.fill}
        stroke={selected ? "#1d4ed8" : colors.stroke}
        strokeWidth={selected ? 3 : 2}
      />
      <text
        x={node.x + NODE_W / 2}
        y={node.y + 1}
        textAnchor="middle"
        dominantBaseline="central"
        className="text-[11px] font-medium select-none pointer-events-none"
        fill={colors.text}
      >
        {node.label.replace(/_/g, " ")}
      </text>
    </g>
  );
}

function EdgeArrow({
  edge,
  nodes,
  maxFreq,
}: {
  edge: PetriNetEdge;
  nodes: PetriNetNode[];
  maxFreq: number;
}) {
  const src = nodes.find((n) => n.id === edge.source);
  const tgt = nodes.find((n) => n.id === edge.target);
  if (!src || !tgt) return null;

  const srcC = nodeCenter(src);
  const tgtC = nodeCenter(tgt);
  const p1 = nodeBorderPoint(src, tgtC.cx, tgtC.cy);
  const p2 = nodeBorderPoint(tgt, srcC.cx, srcC.cy);

  const ratio = maxFreq > 0 ? edge.frequency / maxFreq : 0.5;
  const strokeW = 1 + ratio * 4;
  const opacity = 0.3 + ratio * 0.7;

  const mx = (p1.x + p2.x) / 2;
  const my = (p1.y + p2.y) / 2;

  return (
    <g>
      <line
        x1={p1.x}
        y1={p1.y}
        x2={p2.x}
        y2={p2.y}
        stroke="#64748b"
        strokeWidth={strokeW}
        strokeOpacity={opacity}
        markerEnd="url(#arrowhead)"
      />
      <rect
        x={mx - 14}
        y={my - 9}
        width={28}
        height={18}
        rx={4}
        fill="white"
        fillOpacity={0.85}
        stroke="#cbd5e1"
        strokeWidth={0.5}
      />
      <text
        x={mx}
        y={my + 1}
        textAnchor="middle"
        dominantBaseline="central"
        className="text-[9px] font-semibold select-none pointer-events-none"
        fill="#475569"
      >
        {edge.frequency}
      </text>
    </g>
  );
}

export default function PetriNetViewer() {
  const [graphId, setGraphId] = useState("G1");
  const [data, setData] = useState<PetriNetResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 80, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });
  const svgRef = useRef<SVGSVGElement>(null);

  const fetchGraph = useCallback(async (gid: string) => {
    setLoading(true);
    setError(null);
    setSelectedNode(null);
    try {
      const result = await getPetriNet(gid);
      setData(result);
      if (result.nodes.length > 0) {
        const minY = Math.min(...result.nodes.map((n) => n.y));
        const maxY = Math.max(...result.nodes.map((n) => n.y));
        setPan({ x: 80, y: -(minY + maxY) / 2 });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGraph(graphId);
  }, [graphId, fetchGraph]);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setZoom((z) => Math.max(0.3, Math.min(3, z - e.deltaY * 0.001)));
    };
    svg.addEventListener("wheel", handler, { passive: false });
    return () => svg.removeEventListener("wheel", handler);
  }, [data]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      setDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    },
    [pan],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging) return;
      setPan({
        x: dragStart.current.panX + (e.clientX - dragStart.current.x) / zoom,
        y: dragStart.current.panY + (e.clientY - dragStart.current.y) / zoom,
      });
    },
    [dragging, zoom],
  );

  const handleMouseUp = useCallback(() => setDragging(false), []);

  const selectedInfo = data?.nodes.find((n) => n.id === selectedNode);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50">
          Petri Net Viewer
        </h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
          Directly-Follows Graph mined from event log traces. Edge labels show transition frequency.
        </p>
      </div>

      <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-6">
        <div className="flex items-center gap-4 flex-wrap">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Select Graph
            </label>
            <select
              value={graphId}
              onChange={(e) => setGraphId(e.target.value)}
              className="rounded-md border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-2 text-sm text-zinc-900 dark:text-zinc-100 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none"
            >
              {(data?.available_graphs || ["G1", "G2"]).map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </div>

          {data && (
            <div className="flex items-center gap-4 text-sm text-zinc-500 dark:text-zinc-400">
              <span>
                <strong className="text-zinc-700 dark:text-zinc-200">
                  {data.case_count}
                </strong>{" "}
                traces
              </span>
              <span>
                <strong className="text-zinc-700 dark:text-zinc-200">
                  {data.nodes.length}
                </strong>{" "}
                activities
              </span>
              <span>
                <strong className="text-zinc-700 dark:text-zinc-200">
                  {data.edges.length}
                </strong>{" "}
                flows
              </span>
              <span>
                Total cost:{" "}
                <strong className="text-zinc-700 dark:text-zinc-200">
                  {data.total_cost.toLocaleString()}
                </strong>
              </span>
            </div>
          )}

          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setZoom((z) => Math.min(3, z + 0.2))}
              className="rounded border border-zinc-300 dark:border-zinc-700 px-2 py-1 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-700 dark:text-zinc-300"
            >
              +
            </button>
            <span className="text-xs text-zinc-500 dark:text-zinc-400 w-12 text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom((z) => Math.max(0.3, z - 0.2))}
              className="rounded border border-zinc-300 dark:border-zinc-700 px-2 py-1 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-700 dark:text-zinc-300"
            >
              -
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
        </div>
      )}

      <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-96 text-zinc-400">
            Loading graph...
          </div>
        ) : data ? (
          <div className="flex">
            <svg
              ref={svgRef}
              width="100%"
              height="500"
              className="flex-1 bg-zinc-50 dark:bg-zinc-950 select-none"
              style={{ cursor: dragging ? "grabbing" : "grab" }}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
                </marker>
              </defs>
              <g transform={`translate(${pan.x * zoom + 40}, ${250 + pan.y * zoom}) scale(${zoom})`}>
                {(() => {
                  const maxF = Math.max(...data.edges.map((e) => e.frequency), 1);
                  return data.edges.map((e, i) => (
                    <EdgeArrow key={i} edge={e} nodes={data.nodes} maxFreq={maxF} />
                  ));
                })()}
                {data.nodes.map((n) => (
                  <NodeShape
                    key={n.id}
                    node={n}
                    selected={selectedNode === n.id}
                    onSelect={setSelectedNode}
                  />
                ))}
              </g>
            </svg>

            {selectedInfo && (
              <div className="w-64 border-l border-zinc-200 dark:border-zinc-800 p-4 bg-white dark:bg-zinc-900 shrink-0">
                <h4 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50 mb-3">
                  Node Details
                </h4>
                <dl className="space-y-2 text-sm">
                  {[
                    { label: "Activity", value: selectedInfo.label.replace(/_/g, " ") },
                    { label: "Type", value: selectedInfo.type },
                    { label: "Avg Cost", value: selectedInfo.cost },
                    { label: "Occurrences", value: selectedInfo.human_res },
                  ].map((item) => (
                    <div key={item.label}>
                      <dt className="text-zinc-500 dark:text-zinc-400 text-xs">
                        {item.label}
                      </dt>
                      <dd className="text-zinc-900 dark:text-zinc-100 font-medium">
                        {item.value}
                      </dd>
                    </div>
                  ))}
                </dl>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="mt-4 text-xs text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
                >
                  Close
                </button>
              </div>
            )}
          </div>
        ) : null}
      </div>

      <div className="flex items-center gap-6 text-xs text-zinc-500 dark:text-zinc-400 flex-wrap">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-4 h-4 rounded-full bg-emerald-100 border-2 border-emerald-600" />
          Start
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-4 h-4 rounded bg-indigo-100 border-2 border-indigo-600" />
          Activity
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-4 h-4 rounded-full bg-red-100 border-2 border-red-600" />
          End
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-6 h-0.5 bg-slate-500" />
          <span className="text-[10px] bg-white border border-slate-300 px-1 rounded">n</span>
          Flow (frequency)
        </span>
        <span className="text-zinc-400 dark:text-zinc-600 ml-2">
          Scroll to zoom &middot; Drag to pan &middot; Click node for details
        </span>
      </div>
    </div>
  );
}
