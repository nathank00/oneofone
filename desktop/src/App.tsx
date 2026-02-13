import { useEffect, useState } from "react";
import type { Tab } from "./lib/types";
import { loadSettings } from "./lib/commands";
import Header from "./components/Header";
import Dashboard from "./components/Dashboard";
import ManualMode from "./components/ManualMode";
import AutoMode from "./components/AutoMode";
import Settings from "./components/Settings";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("home");
  const [useDemoApi, setUseDemoApi] = useState(true);

  useEffect(() => {
    loadSettings()
      .then((s) => setUseDemoApi(s.useDemoApi))
      .catch(() => {});
  }, [activeTab]);

  return (
    <div className="flex min-h-screen flex-col bg-neutral-950 text-neutral-100">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        useDemoApi={useDemoApi}
      />
      <main className="flex-1">
        {activeTab === "home" && (
          <Dashboard onNavigate={setActiveTab} />
        )}
        {activeTab === "manual" && <ManualMode />}
        {activeTab === "auto" && <AutoMode />}
        {activeTab === "settings" && <Settings />}
      </main>
    </div>
  );
}
