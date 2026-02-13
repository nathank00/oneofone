import type { GameLog } from "@/lib/types";

interface PredictionCardProps {
  game: GameLog;
}

/**
 * Extract the common short name for an NBA team.
 * Handles edge cases like "Trail Blazers" and "76ers".
 */
function shortName(fullName: string): string {
  if (fullName.includes("Trail Blazers")) return "Blazers";
  if (fullName.includes("76ers")) return "76ers";
  const parts = fullName.split(" ");
  return parts[parts.length - 1];
}

export default function PredictionCard({ game }: PredictionCardProps) {
  const {
    AWAY_NAME,
    HOME_NAME,
    PREDICTION,
    PREDICTION_PCT,
    GAME_OUTCOME,
    GAME_STATUS,
    AWAY_PTS,
    HOME_PTS,
  } = game;

  const hasPrediction = PREDICTION !== null && PREDICTION_PCT !== null;
  const isCompleted = GAME_STATUS === 4 || GAME_OUTCOME !== null;

  // Predicted winner name
  const predictedWinner = PREDICTION === 1 ? HOME_NAME : AWAY_NAME;

  // Confidence from the predicted winner's perspective
  const confidence =
    hasPrediction
      ? PREDICTION === 1
        ? PREDICTION_PCT!
        : 1 - PREDICTION_PCT!
      : null;

  const confidenceStr =
    confidence !== null ? `${Math.round(confidence * 100)}%` : null;

  // Check if prediction was correct (only for finished games with predictions)
  let predictionCorrect: boolean | null = null;
  if (isCompleted && hasPrediction && GAME_OUTCOME !== null) {
    predictionCorrect = PREDICTION === GAME_OUTCOME;
  }

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4 transition-colors hover:border-neutral-700">
      {/* Matchup */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span
            className={`text-sm font-medium truncate ${
              PREDICTION === 0 ? "text-white" : "text-neutral-400"
            }`}
          >
            {AWAY_NAME}
          </span>
          <span className="text-neutral-600 shrink-0">@</span>
          <span
            className={`text-sm font-medium truncate ${
              PREDICTION === 1 ? "text-white" : "text-neutral-400"
            }`}
          >
            {HOME_NAME}
          </span>
        </div>

        {/* Score for completed games */}
        {isCompleted && AWAY_PTS != null && HOME_PTS != null && (
          <span className="font-mono text-sm text-neutral-500 ml-3 shrink-0">
            {AWAY_PTS} - {HOME_PTS}
          </span>
        )}
      </div>

      {/* Prediction details */}
      {hasPrediction ? (
        <div className="mt-2 flex items-center gap-3">
          <span className="text-xs uppercase tracking-wider text-neutral-600">
            Pick
          </span>
          <span className="text-sm font-semibold text-white">
            {shortName(predictedWinner)}
          </span>
          <span className="font-mono text-sm text-neutral-400">
            {confidenceStr}
          </span>

          {/* W/L badge for finished games */}
          {predictionCorrect !== null && (
            <span
              className={`ml-auto rounded-full px-2.5 py-0.5 text-xs font-bold ${
                predictionCorrect
                  ? "bg-green-900/40 text-green-400"
                  : "bg-red-900/40 text-red-400"
              }`}
            >
              {predictionCorrect ? "W" : "L"}
            </span>
          )}
        </div>
      ) : (
        <div className="mt-2">
          <span className="text-xs text-neutral-600">No prediction</span>
        </div>
      )}
    </div>
  );
}
