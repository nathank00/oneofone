import { useEffect, useRef } from "react";
import type { ScannerEvent } from "../lib/types";

interface EventLogProps {
  events: ScannerEvent[];
}

const typeStyles: Record<string, string> = {
  scan: "text-neutral-500",
  edge_found: "text-emerald-400",
  bet_placed: "text-green-400",
  error: "text-red-400",
};

const typeLabels: Record<string, string> = {
  scan: "SCAN",
  edge_found: "EDGE",
  bet_placed: "BET",
  error: "ERR",
};

export default function EventLog({ events }: EventLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-6 text-center text-sm text-neutral-600">
        No events yet. Start the scanner to begin.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 max-h-[420px] overflow-y-auto">
      <div className="divide-y divide-neutral-800/60">
        {events.map((event, i) => {
          const style = typeStyles[event.eventType] || "text-neutral-400";
          const label = typeLabels[event.eventType] || event.eventType.toUpperCase();

          return (
            <div
              key={i}
              className="flex items-start gap-3 px-4 py-2.5 text-sm"
            >
              <span className="shrink-0 font-mono text-[10px] text-neutral-600 pt-0.5">
                {event.timestamp.split("T").pop()?.split(".")[0] || event.timestamp}
              </span>
              <span
                className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold tracking-wider ${style} bg-neutral-800/60`}
              >
                {label}
              </span>
              <span className="text-neutral-300 min-w-0 break-words">
                {event.message}
              </span>
              {event.edge !== null && (
                <span className="shrink-0 font-mono text-xs text-emerald-400">
                  +{event.edge.toFixed(1)}%
                </span>
              )}
              {event.orderId && (
                <span className="shrink-0 font-mono text-[10px] text-neutral-500">
                  #{event.orderId.slice(0, 8)}
                </span>
              )}
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
