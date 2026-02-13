import { useEffect, useState } from "react";
import {
  getPredictions,
  getTodaysPredictionsRaw,
  loadSettings,
} from "../lib/commands";
import type { AppSettings, MatchedGame, PredictionDisplay } from "../lib/types";
import GameRow from "./GameRow";

/** Wrap a promise with a timeout so the UI never hangs if the backend panics. */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), ms)
    ),
  ]);
}

export default function ManualMode() {
  // ── Section 1 state: predictions from Supabase DB ──
  const [predictions, setPredictions] = useState<PredictionDisplay[]>([]);
  const [predsLoading, setPredsLoading] = useState(true);
  const [predsError, setPredsError] = useState<string | null>(null);

  // ── Section 2 state: Kalshi markets ──
  const [markets, setMarkets] = useState<MatchedGame[]>([]);
  const [marketsLoading, setMarketsLoading] = useState(true);
  const [marketsError, setMarketsError] = useState<string | null>(null);

  // ── Settings (for bet sizing) ──
  const [settings, setCurrentSettings] = useState<AppSettings | null>(null);

  // ── Fetch predictions from our Supabase DB (no Kalshi auth needed) ──
  const fetchPredictions = async () => {
    setPredsLoading(true);
    setPredsError(null);
    try {
      const rawPreds = await withTimeout(getTodaysPredictionsRaw(), 10000);
      setPredictions(rawPreds);
    } catch (e) {
      setPredsError(String(e));
      setPredictions([]);
    }
    setPredsLoading(false);
  };

  // ── Fetch matched markets from Kalshi (needs auth) ──
  const fetchMarkets = async () => {
    setMarketsLoading(true);
    setMarketsError(null);
    try {
      const data = await withTimeout(getPredictions(), 15000);
      setMarkets(data);
    } catch (e) {
      setMarketsError(String(e));
      setMarkets([]);
    }
    setMarketsLoading(false);
  };

  useEffect(() => {
    // Load settings for bet sizing
    withTimeout(loadSettings(), 5000)
      .then(setCurrentSettings)
      .catch(() => {});

    // Fetch both sections independently & in parallel
    fetchPredictions();
    fetchMarkets();
  }, []);

  const refreshAll = () => {
    fetchPredictions();
    fetchMarkets();
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="font-mono text-xl font-bold tracking-wider text-white">
          Manual Trading
        </h1>
        <button
          onClick={refreshAll}
          disabled={predsLoading && marketsLoading}
          className="rounded-md border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition-all hover:border-neutral-500 hover:text-white disabled:opacity-50"
        >
          {predsLoading && marketsLoading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {/* ─────────────────── SECTION 1: Today's Predictions (from DB) ─────────────────── */}
      <div>
        <h2 className="text-xs uppercase tracking-wider text-neutral-500 font-semibold mb-3">
          Today's Predictions
          <span className="text-neutral-600 font-normal ml-1">— from model</span>
        </h2>

        {predsError && (
          <div className="mb-3 rounded-md border border-red-800/40 bg-red-900/20 p-3 text-sm text-red-400">
            {predsError}
          </div>
        )}

        {predsLoading ? (
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="h-16 animate-pulse rounded-lg bg-neutral-800/50"
              />
            ))}
          </div>
        ) : predictions.length === 0 ? (
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 py-10 text-center text-sm text-neutral-600">
            No predictions for today. Games will appear here once the model runs.
          </div>
        ) : (
          <div className="space-y-2">
            {predictions.map((pred) => (
              <div
                key={pred.gameId}
                className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-neutral-300">
                      {pred.awayName}
                    </span>
                    <span className="text-neutral-600">@</span>
                    <span className="text-sm font-medium text-neutral-300">
                      {pred.homeName}
                    </span>
                    {pred.gameStatus === 1 && (
                      <span className="rounded bg-blue-900/40 border border-blue-800/30 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-blue-400">
                        Upcoming
                      </span>
                    )}
                    {pred.gameStatus === 2 && (
                      <span className="rounded bg-green-900/40 border border-green-800/30 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-green-400">
                        Live
                      </span>
                    )}
                    {pred.gameStatus === 3 && (
                      <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-neutral-500">
                        Final
                      </span>
                    )}
                  </div>
                </div>
                <div className="mt-2 flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-neutral-600">
                      Pick
                    </span>
                    <span className="font-semibold text-white">
                      {pred.predictedWinner.split(" ").pop()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider text-neutral-600">
                      Model
                    </span>
                    <span className="font-mono text-white">
                      {Math.round(pred.winProbability * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ─────────────────── SECTION 2: Kalshi Markets (tradeable) ─────────────────── */}
      <div>
        <h2 className="text-xs uppercase tracking-wider text-neutral-500 font-semibold mb-3">
          Kalshi Markets
          <span className="text-neutral-600 font-normal ml-1">— place bets</span>
        </h2>

        {marketsError && (
          <div className="mb-3 rounded-md border border-red-800/40 bg-red-900/20 p-3 text-sm text-red-400">
            {marketsError}
          </div>
        )}

        {marketsLoading ? (
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="h-20 animate-pulse rounded-lg bg-neutral-800/50"
              />
            ))}
          </div>
        ) : markets.length === 0 ? (
          <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 py-10 text-center text-sm text-neutral-600">
            No live NBA markets on Kalshi right now.
          </div>
        ) : (
          <div className="space-y-2">
            {markets.map((game) => (
              <GameRow
                key={game.marketTicker}
                game={game}
                sizingMode={settings?.sizingMode ?? "contracts"}
                betAmount={settings?.betAmount ?? 10}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
