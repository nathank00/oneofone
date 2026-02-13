import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  startScanner,
  stopScanner,
  getScannerStatus,
  getTodaysPredictionsRaw,
  loadSettings,
} from "../lib/commands";
import type { PredictionDisplay, ScannerEvent, SizingMode } from "../lib/types";
import EventLog from "./EventLog";

/** Wrap a promise with a timeout so the UI never hangs if the backend panics. */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), ms)
    ),
  ]);
}

export default function AutoMode() {
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<ScannerEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [edgeThreshold, setEdgeThreshold] = useState(10);
  const [betAmount, setBetAmount] = useState(10);
  const [sizingMode, setSizingMode] = useState<SizingMode>("contracts");
  const [predictions, setPredictions] = useState<PredictionDisplay[]>([]);

  // Stats computed from events
  const betsPlaced = events.filter((e) => e.eventType === "bet_placed").length;
  const edgesFound = events.filter((e) => e.eventType === "edge_found").length;
  const errors = events.filter((e) => e.eventType === "error").length;

  useEffect(() => {
    // Load current settings to show in UI
    withTimeout(loadSettings(), 5000)
      .then((s) => {
        setEdgeThreshold(s.edgeThreshold);
        setBetAmount(s.betAmount);
        setSizingMode(s.sizingMode);
      })
      .catch(() => {});

    // Fetch today's predictions
    withTimeout(getTodaysPredictionsRaw(), 10000)
      .then(setPredictions)
      .catch(() => {});

    // Get current scanner status
    withTimeout(getScannerStatus(), 5000)
      .then(setRunning)
      .catch(() => {});

    // Listen for scanner events from the Rust backend
    const unlisten = listen<ScannerEvent>("scanner-event", (event) => {
      setEvents((prev) => [...prev, event.payload]);
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const handleToggle = async () => {
    setError(null);
    try {
      if (running) {
        await withTimeout(stopScanner(), 5000);
        setRunning(false);
      } else {
        await withTimeout(startScanner(), 5000);
        setRunning(true);
      }
    } catch (e) {
      setError(String(e));
    }
  };

  const clearLog = () => setEvents([]);

  return (
    <div className="p-6 space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="font-mono text-xl font-bold tracking-wider text-white">
          Auto Scanner
        </h1>
        <button
          onClick={clearLog}
          className="text-xs text-neutral-600 hover:text-neutral-400 transition-colors"
        >
          Clear Log
        </button>
      </div>

      {/* Toggle + Status */}
      <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={handleToggle}
              className={`relative h-8 w-14 rounded-full transition-colors ${
                running ? "bg-green-700" : "bg-neutral-700"
              }`}
            >
              <span
                className={`absolute top-1 left-1 h-6 w-6 rounded-full bg-white transition-transform ${
                  running ? "translate-x-6" : "translate-x-0"
                }`}
              />
            </button>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-white">
                  {running ? "Scanner Active" : "Scanner Off"}
                </span>
                {running && (
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500" />
                  </span>
                )}
              </div>
              <span className="text-xs text-neutral-500">
                {running
                  ? "Scanning every 30s for edges above threshold"
                  : "Toggle on to start auto-scanning"}
              </span>
            </div>
          </div>

          {/* Config summary */}
          <div className="flex items-center gap-4 text-xs text-neutral-500">
            <div>
              <span className="uppercase tracking-wider">Threshold</span>{" "}
              <span className="font-mono text-neutral-300">{edgeThreshold}%</span>
            </div>
            <div>
              <span className="uppercase tracking-wider">Size</span>{" "}
              <span className="font-mono text-neutral-300">
                {sizingMode === "dollars"
                  ? `$${betAmount}`
                  : `${betAmount} contracts`}
              </span>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-red-800/40 bg-red-900/20 p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Session stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-3 text-center">
          <div className="font-mono text-lg font-bold text-emerald-400">
            {edgesFound}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">
            Edges Found
          </div>
        </div>
        <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-3 text-center">
          <div className="font-mono text-lg font-bold text-green-400">
            {betsPlaced}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">
            Bets Placed
          </div>
        </div>
        <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 p-3 text-center">
          <div className="font-mono text-lg font-bold text-red-400">
            {errors}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">
            Errors
          </div>
        </div>
      </div>

      {/* Today's predictions */}
      {predictions.length > 0 && (
        <div>
          <h2 className="text-xs uppercase tracking-wider text-neutral-500 font-semibold mb-2">
            Today's Predictions ({predictions.length})
          </h2>
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 divide-y divide-neutral-800/60">
            {predictions.map((pred) => (
              <div key={pred.gameId} className="flex items-center justify-between px-4 py-2.5">
                <div className="flex items-center gap-2 text-sm min-w-0">
                  <span className="text-neutral-400 truncate">{pred.awayName}</span>
                  <span className="text-neutral-600 shrink-0">@</span>
                  <span className="text-neutral-400 truncate">{pred.homeName}</span>
                  {pred.gameStatus === 1 && (
                    <span className="rounded bg-blue-900/40 border border-blue-800/30 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-blue-400 shrink-0">
                      Upcoming
                    </span>
                  )}
                  {pred.gameStatus === 2 && (
                    <span className="rounded bg-green-900/40 border border-green-800/30 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-green-400 shrink-0">
                      Live
                    </span>
                  )}
                  {pred.gameStatus === 3 && (
                    <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-neutral-500 shrink-0">
                      Final
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <span className="text-xs font-semibold text-white">
                    {pred.predictedWinner.split(" ").pop()}
                  </span>
                  <span className="font-mono text-xs text-neutral-300">
                    {Math.round(pred.winProbability * 100)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Event log */}
      <div>
        <h2 className="text-xs uppercase tracking-wider text-neutral-500 font-semibold mb-2">
          Event Log
        </h2>
        <EventLog events={events} />
      </div>
    </div>
  );
}
