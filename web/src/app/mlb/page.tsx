import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "[ ONE OF ONE ] — MLB",
};

export default function MlbPage() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
      <h2 className="mb-4 font-mono text-3xl font-bold tracking-wider text-neutral-400">
        MLB
      </h2>
      <p className="text-neutral-600">Coming soon.</p>
    </div>
  );
}
