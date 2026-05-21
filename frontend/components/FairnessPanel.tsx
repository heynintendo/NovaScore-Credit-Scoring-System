"use client";

import { motion } from "motion/react";

interface Props {
  rawScore: number;
  adjustedScore: number;
  reason: string | null;
  ageBucket: string;
}

export function FairnessPanel({ rawScore, adjustedScore, reason, ageBucket }: Props) {
  const delta = adjustedScore - rawScore;
  const hasAdj = Math.abs(delta) >= 0.5;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1], delay: 0.05 }}
      className="surface-edge rounded-2xl p-7"
    >
      <div className="flex items-baseline justify-between gap-4">
        <p className="label-mono">Fairness audit</p>
        <span className="font-mono text-[11px] tabular-nums tracking-widest text-cream-muted">
          Age bucket · {ageBucket}
        </span>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-5">
        <div>
          <dt className="label-mono-faint">Model</dt>
          <dd className="mt-1 font-display text-2xl text-cream tabular-nums">
            {rawScore.toFixed(0)}
          </dd>
        </div>
        <div>
          <dt className="label-mono-faint">Mitigated</dt>
          <dd className="mt-1 font-display text-2xl text-gold tabular-nums">
            {adjustedScore.toFixed(0)}
          </dd>
        </div>
      </dl>

      {hasAdj ? (
        <p className="mt-5 max-w-prose text-sm leading-relaxed text-cream-muted">
          <span
            className="font-mono text-xs tracking-widest"
            style={{ color: delta > 0 ? "#D4B47A" : "#A8A8B3" }}
          >
            {delta > 0 ? "+" : "−"}
            {Math.abs(delta).toFixed(1)}
          </span>
          {"  "}
          {reason ?? "Per-group equal-opportunity threshold applied."}
        </p>
      ) : (
        <p className="mt-5 text-sm leading-relaxed text-cream-muted">
          No fairness adjustment needed at this score.
        </p>
      )}
    </motion.div>
  );
}
