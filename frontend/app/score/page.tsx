"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { FeatureInput } from "@/components/FeatureInput";
import { ScoreGauge } from "@/components/ScoreGauge";
import { DecisionCard } from "@/components/DecisionCard";
import { FairnessPanel } from "@/components/FairnessPanel";
import { scoreApplicant, SAMPLE_APPLICANT } from "@/lib/api";
import type { ScoreRequest, ScoreResponse } from "@/lib/types";

export default function ScorePage() {
  const [values, setValues] = useState<ScoreRequest>(SAMPLE_APPLICANT);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        setError(null);
        const r = await scoreApplicant(values);
        setResult(r);
      } catch (e) {
        setError((e as Error).message);
      }
    }, 280);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [values]);

  const score = result?.novascore ?? 720;
  const tier = result?.decision_band ?? "Gold";
  const adjusted = result?.fairness_adjusted_score ?? score;

  return (
    <div className="relative pt-32 pb-28 lg:pt-40">
      <div className="mx-auto max-w-7xl px-6 lg:px-10">
        <motion.header
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="mb-16 lg:mb-20"
        >
          <p className="label-mono">The calculator</p>
          <h1 className="mt-3 max-w-3xl font-display text-[clamp(2.5rem,6vw,4.5rem)] font-medium leading-[0.98] text-cream">
            Sketch an applicant.<br />
            <span className="italic text-cream/75">See how the model answers.</span>
          </h1>
          <p className="mt-6 max-w-prose text-cream-muted">
            Live wire-up to a LightGBM model trained on Home Credit Default
            Risk. Adjust any field, watch the gauge spring to a new tier in
            under a second. The fairness audit runs alongside the model.
          </p>
        </motion.header>

        <div className="grid grid-cols-12 gap-x-6 gap-y-12">
          <div className="col-span-12 lg:col-span-7">
            <FeatureInput values={values} setValues={setValues} />
          </div>

          <div className="col-span-12 space-y-5 lg:col-span-5 lg:sticky lg:top-28 lg:self-start">
            <AnimatePresence mode="wait">
              {error ? (
                <motion.div
                  key="err"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="surface-edge rounded-2xl p-6 text-sm text-tier-decline"
                  style={{ color: "#D88B8B" }}
                >
                  <p className="label-mono mb-2" style={{ color: "#D88B8B" }}>
                    Could not reach scoring service
                  </p>
                  <p className="text-pretty leading-relaxed">{error}</p>
                  <p className="mt-3 text-xs text-cream-muted">
                    Set <code className="font-mono text-gold">NEXT_PUBLIC_API_URL</code> to the API base URL.
                  </p>
                </motion.div>
              ) : null}
            </AnimatePresence>

            <ScoreGauge score={score} tier={tier} adjustedScore={adjusted} />
            <DecisionCard tier={tier} />
            <FairnessPanel
              rawScore={score}
              adjustedScore={adjusted}
              reason={result?.fairness_adjustment_reason ?? null}
              ageBucket={result?.age_bucket ?? "26-40"}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
