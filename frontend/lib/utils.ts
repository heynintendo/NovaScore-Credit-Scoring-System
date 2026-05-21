import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** Standard shadcn-style className combiner — clsx + tailwind-merge. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type Tier = "Platinum" | "Gold" | "Silver" | "Bronze";

export function decisionBand(score: number): Tier {
  if (score >= 800) return "Platinum";
  if (score >= 700) return "Gold";
  if (score >= 600) return "Silver";
  return "Bronze";
}

/** Maps score → arc fraction (0–1) on the gauge sweep from 300 to 950. */
export function scoreToFraction(score: number): number {
  const clipped = Math.max(300, Math.min(950, score));
  return (clipped - 300) / (950 - 300);
}

export const TIER_COLOR: Record<Tier, string> = {
  Platinum: "#C9A26F",
  Gold: "#D4B47A",
  Silver: "#A8A8B3",
  Bronze: "#8B6B4A",
};

export const TIER_POLICY: Record<Tier, string> = {
  Platinum: "Auto-approve. Large credit limit, lower interest, premium benefits.",
  Gold: "Standard approval. Medium limit, standard rates, upgrade path available.",
  Silver: "Manual review recommended. Smaller limit, repayment coaching offered.",
  Bronze: "Application declined. Coaching and savings plans available to rebuild standing.",
};

export const TIER_RANGE: Record<Tier, string> = {
  Platinum: "800 – 950",
  Gold: "700 – 799",
  Silver: "600 – 699",
  Bronze: "300 – 599",
};
