interface RecordBadgeProps {
  label: string;
  wins: number;
  losses: number;
}

export default function RecordBadge({ label, wins, losses }: RecordBadgeProps) {
  const total = wins + losses;
  const pct = total > 0 ? ((wins / total) * 100).toFixed(1) : "—";

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3">
      <div className="text-xs uppercase tracking-wider text-neutral-500">
        {label}
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="font-mono text-xl font-bold text-white">
          {wins}-{losses}
        </span>
        {total > 0 && (
          <span className="font-mono text-sm text-neutral-500">{pct}%</span>
        )}
      </div>
    </div>
  );
}
