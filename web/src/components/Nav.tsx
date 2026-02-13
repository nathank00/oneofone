"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/nba", label: "NBA" },
  { href: "/mlb", label: "MLB" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-neutral-800 bg-neutral-950/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-4">
        <Link
          href="/"
          className="font-mono text-lg tracking-widest text-neutral-100 hover:text-white transition-colors"
        >
          [ ONE OF ONE ]
        </Link>
        <div className="flex gap-6">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-sm font-medium uppercase tracking-wider transition-colors ${
                pathname.startsWith(link.href)
                  ? "text-white"
                  : "text-neutral-500 hover:text-neutral-300"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
