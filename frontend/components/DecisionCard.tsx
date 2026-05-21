"use client";

import { motion } from "motion/react";
import type { Tier } from "@/lib/utils";
import { TIER_COLOR, TIER_POLICY, TIER_RANGE } from "@/lib/utils";

interface Props {
  tier: Tier;
}

export function DecisionCard({ tier }: Props) {
  return (
    <motion.div
      key={tier}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      className="surface-edge rounded-2xl p-7"
    >
      <div className="flex items-baseline justify-between gap-4">
        <p className="label-mono">Decision</p>
        <span className="font-mono text-[11px] tabular-nums tracking-widest text-cream-muted">
          {TIER_RANGE[tier]}
        </span>
      </div>

      <div className="mt-4 flex items-center gap-4">
        <span
          className="block h-2 w-2 rounded-full"
          style={{ background: TIER_COLOR[tier], boxShadow: `0 0 16px ${TIER_COLOR[tier]}` }}
        />
        <span
          className="font-display text-3xl italic"
          style={{ color: TIER_COLOR[tier] }}
        >
          {tier}
        </span>
      </div>

      <p className="mt-5 text-pretty leading-relaxed text-cream-muted">
        {TIER_POLICY[tier]}
      </p>
    </motion.div>
  );
}
