import type { Metadata } from "next";
import { Fraunces, Geist, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Navigation } from "@/components/Navigation";

const fraunces = Fraunces({
  variable: "--font-display",
  subsets: ["latin"],
  axes: ["opsz"],
  display: "swap",
});

const geist = Geist({
  variable: "--font-body",
  subsets: ["latin"],
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "NovaScore — Equitable credit scoring",
  description:
    "An equitable credit scoring engine demonstrated on the Home Credit Default Risk public dataset. Calibrated 300–950 score, threshold-based fairness mitigation, 0.75 test AUROC.",
  metadataBase: new URL("https://novascore.dev"),
  openGraph: {
    title: "NovaScore",
    description: "Respecting those who carry trust, with credit that carries them.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${fraunces.variable} ${geist.variable} ${jetbrains.variable}`}>
      <body>
        <Navigation />
        <main>{children}</main>
        <footer className="relative z-10 mt-32 border-t border-white/[0.05] py-12">
          <div className="mx-auto max-w-7xl px-6 lg:px-10">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="label-mono text-cream-muted">NovaScore · v0.2 · BIT Mesra</p>
                <p className="mt-3 text-sm text-cream-muted">
                  Built by Anish Kishore.
                </p>
              </div>
              <a
                href="https://github.com/heynintendo/NovaScore-Credit-Scoring-System"
                className="text-sm text-cream-muted underline decoration-cream/30 underline-offset-4 transition-colors hover:text-cream hover:decoration-gold"
              >
                github.com/heynintendo/NovaScore-Credit-Scoring-System
              </a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
