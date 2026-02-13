import { useEffect, useState } from "react";
import {
  loadSettings,
  saveSettings,
  pickPemFile,
  testKalshiConnection,
} from "../lib/commands";
import type { AppSettings, SizingMode } from "../lib/types";

export default function Settings() {
  const [settings, setSettings] = useState<AppSettings>({
    kalshiKeyId: "",
    kalshiPemPath: "",
    edgeThreshold: 10,
    sizingMode: "contracts",
    betAmount: 10,
    useDemoApi: true,
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testLoading, setTestLoading] = useState(false);

  useEffect(() => {
    loadSettings()
      .then(setSettings)
      .catch(() => {});
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    try {
      await saveSettings(settings);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      console.error(e);
    }
    setSaving(false);
  };

  const handleTest = async () => {
    setTestLoading(true);
    setTestResult(null);
    try {
      const result = await testKalshiConnection();
      setTestResult(result);
    } catch (e) {
      setTestResult(`Error: ${e}`);
    }
    setTestLoading(false);
  };

  const handlePickPem = async () => {
    try {
      const path = await pickPemFile();
      if (path) {
        setSettings((s) => ({ ...s, kalshiPemPath: path }));
      }
    } catch (e) {
      console.error("File picker error:", e);
    }
  };

  const update = (field: keyof AppSettings, value: string | number | boolean) => {
    setSettings((s) => ({ ...s, [field]: value }));
  };

  const isContracts = settings.sizingMode === "contracts";

  return (
    <div className="p-6 space-y-6 max-w-lg">
      <h1 className="font-mono text-xl font-bold tracking-wider text-white">
        Settings
      </h1>

      {/* Kalshi API */}
      <section className="space-y-3">
        <h2 className="text-xs uppercase tracking-wider text-neutral-500 font-semibold">
          Kalshi API
        </h2>
        <div>
          <label className="block text-xs text-neutral-400 mb-1">Key ID</label>
          <input
            type="text"
            value={settings.kalshiKeyId}
            onChange={(e) => update("kalshiKeyId", e.target.value)}
            placeholder="Your Kalshi API Key ID"
            className="w-full rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 outline-none focus:border-neutral-500"
          />
        </div>
        <div>
          <label className="block text-xs text-neutral-400 mb-1">
            Private Key PEM File
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={settings.kalshiPemPath}
              onChange={(e) => update("kalshiPemPath", e.target.value)}
              placeholder="/path/to/kalshi-private-key.pem"
              className="flex-1 rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 outline-none focus:border-neutral-500"
            />
            <button
              onClick={handlePickPem}
              className="shrink-0 rounded-md border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-neutral-300 transition-all hover:border-neutral-500 hover:text-white"
            >
              Browse
            </button>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.useDemoApi}
              onChange={(e) => update("useDemoApi", e.target.checked)}
              className="rounded accent-amber-500"
            />
            <span className="text-sm text-neutral-300">Use Demo API</span>
          </label>
          <span className="text-xs text-neutral-600">
            {settings.useDemoApi ? "(sandbox — no real money)" : "(PRODUCTION — real money!)"}
          </span>
        </div>
      </section>

      {/* Trading */}
      <section className="space-y-3">
        <h2 className="text-xs uppercase tracking-wider text-neutral-500 font-semibold">
          Trading
        </h2>

        {/* Edge threshold */}
        <div>
          <label className="block text-xs text-neutral-400 mb-1">
            Edge Threshold (%)
          </label>
          <input
            type="number"
            min={1}
            max={30}
            step={1}
            value={settings.edgeThreshold}
            onChange={(e) => update("edgeThreshold", Number(e.target.value))}
            className="w-full rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm text-neutral-100 outline-none focus:border-neutral-500"
          />
        </div>

        {/* Position sizing mode */}
        <div>
          <label className="block text-xs text-neutral-400 mb-2">
            Position Sizing
          </label>
          <div className="flex rounded-md border border-neutral-700 overflow-hidden">
            <button
              onClick={() => update("sizingMode", "contracts" as SizingMode)}
              className={`flex-1 py-2 text-sm font-medium transition-colors ${
                isContracts
                  ? "bg-neutral-700 text-white"
                  : "bg-neutral-900 text-neutral-500 hover:text-neutral-300"
              }`}
            >
              By Contracts
            </button>
            <button
              onClick={() => update("sizingMode", "dollars" as SizingMode)}
              className={`flex-1 py-2 text-sm font-medium transition-colors ${
                !isContracts
                  ? "bg-neutral-700 text-white"
                  : "bg-neutral-900 text-neutral-500 hover:text-neutral-300"
              }`}
            >
              By Dollars
            </button>
          </div>
          <p className="mt-1 text-[11px] text-neutral-600">
            {isContracts
              ? "Buy a fixed number of contracts per bet"
              : "Spend up to this dollar amount per bet (buys as many contracts as possible without exceeding it)"}
          </p>
        </div>

        {/* Bet amount */}
        <div>
          <label className="block text-xs text-neutral-400 mb-1">
            {isContracts ? "Contracts per Bet" : "Dollars per Bet"}
          </label>
          <div className="relative">
            {!isContracts && (
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm text-neutral-500">
                $
              </span>
            )}
            <input
              type="number"
              min={isContracts ? 1 : 1}
              max={isContracts ? 1000 : 10000}
              step={isContracts ? 1 : 1}
              value={settings.betAmount}
              onChange={(e) => {
                const val = Number(e.target.value);
                if (val >= (isContracts ? 1 : 1)) {
                  update("betAmount", val);
                }
              }}
              className={`w-full rounded-md border border-neutral-700 bg-neutral-900 py-2 text-sm text-neutral-100 outline-none focus:border-neutral-500 ${
                !isContracts ? "pl-7 pr-3" : "px-3"
              }`}
            />
          </div>
        </div>
      </section>

      {/* Actions */}
      <div className="flex items-center gap-3 pt-2">
        <button
          onClick={handleSave}
          disabled={saving}
          className="rounded-lg border border-neutral-600 bg-neutral-800 px-5 py-2 text-sm font-medium text-white transition-all hover:border-neutral-500 hover:bg-neutral-700 disabled:opacity-50"
        >
          {saving ? "Saving..." : saved ? "Saved ✓" : "Save Settings"}
        </button>
        <button
          onClick={handleTest}
          disabled={testLoading}
          className="rounded-lg border border-neutral-700 bg-neutral-900 px-5 py-2 text-sm font-medium text-neutral-300 transition-all hover:border-neutral-500 hover:text-white disabled:opacity-50"
        >
          {testLoading ? "Testing..." : "Test Connection"}
        </button>
      </div>

      {testResult && (
        <div
          className={`rounded-md p-3 text-sm ${
            testResult.startsWith("Error")
              ? "bg-red-900/20 text-red-400 border border-red-800/40"
              : "bg-green-900/20 text-green-400 border border-green-800/40"
          }`}
        >
          {testResult}
        </div>
      )}
    </div>
  );
}
