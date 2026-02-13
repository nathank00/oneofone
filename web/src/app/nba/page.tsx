import type { Metadata } from "next";
import NbaDashboard from "@/components/NbaDashboard";

export const metadata: Metadata = {
  title: "[ ONE OF ONE ] — NBA",
};

export default function NbaPage() {
  return <NbaDashboard />;
}
