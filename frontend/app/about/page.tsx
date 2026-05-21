"use client";

import { motion } from "motion/react";
import Link from "next/link";

export default function AboutPage() {
  return (
    <article className="relative pb-32 pt-32 lg:pt-40">
      <div className="mx-auto max-w-3xl px-6 lg:px-10">
        <motion.header
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          <p className="label-mono">Methodology</p>
          <h1 className="mt-3 font-display text-[clamp(2.75rem,6vw,4.75rem)] font-medium leading-[0.98] text-cream">
            How it works,<br />
            <span className="italic">in full sentences.</span>
          </h1>
        </motion.header>

        <Section eyebrow="01" title="Dataset journey">
          <p>
            NovaScore was originally designed around synthetic gig-worker
            behavioral data. After 15 hyperparameter trials, the synthetic
            generator&apos;s signal-to-noise plateau capped AUROC at{" "}
            <Hi>~0.61</Hi> regardless of architecture. We pivoted to the
            Home Credit Default Risk public dataset —{" "}
            <Hi>307,511 real anonymized loan applications</Hi> — for credible
            benchmarking. The synthetic generator is preserved in the repo
            for reproducibility and ablation comparison.
          </p>
          <p>
            Honest reporting of what works — and what doesn&apos;t — is part of
            the methodology.
          </p>
        </Section>

        <Section eyebrow="02" title="The 3-L framework">
          <p>
            Three loops in sequence. <Hi>Learn</Hi> turns raw signals into
            features. <Hi>Lend</Hi> turns features into a calibrated score.{" "}
            <Hi>Loop</Hi> audits the resulting decisions for group-level
            harm and corrects via per-group threshold optimization. Each
            loop is independently testable; nothing in the pipeline depends
            on the others except through serialized artifacts.
          </p>
        </Section>

        <Section eyebrow="03" title="Data">
          <p>
            Primary application table with ~120 features, augmented by
            per-applicant aggregates from <Hi>bureau</Hi>,{" "}
            <Hi>previous_application</Hi>,{" "}
            <Hi>POS_CASH_balance</Hi>, and{" "}
            <Hi>installments_payments</Hi>. Sequence features —{" "}
            60-month bureau_balance status histories one-hot encoded into
            eight channels — feed the TCN branch. SelectKBest with
            f_classif narrows the engineered 165 columns to the most
            predictive 80, so the FT-Transformer&apos;s O(n²) attention stays
            tractable on commodity hardware.
          </p>
        </Section>

        <Section eyebrow="04" title="Architecture">
          <p>
            A LightGBM baseline (test AUROC <Hi>0.7450</Hi>) runs in
            parallel to a hybrid model: an FT-Transformer (per-feature
            tokenization, 2-layer encoder) over tabular features,
            concatenated with a 3-block dilated TCN over the bureau
            sequence. The two predictions are blended via a
            val-AUROC-weighted average into a single ensemble.
          </p>
          <ArchitectureSketch />
        </Section>

        <Section eyebrow="05" title="Results">
          <Results />
          <p className="mt-6">
            LightGBM with proper feature engineering remained the production
            workhorse; the FT-Transformer + TCN hybrid demonstrated
            multimodal architectural exploration but did not surpass
            tree-based methods on this distribution. The ensemble combined
            both for marginal gain (<Hi>+0.0006</Hi> over LightGBM alone on
            test).
          </p>
        </Section>

        <Section eyebrow="06" title="Fairness analysis">
          <p>
            ΔTPR on age bucket: <Hi>0.3229 → 0.0044</Hi> via per-group
            threshold optimization that equalises TPR to ~0.80 across all
            four age buckets. Per-group score adjustments at the decision
            boundary:
          </p>
          <FairnessTable />
          <p className="mt-6">
            The intervention boosts <Hi>18-25</Hi> at the decision threshold
            (whose model TPR was already very high at 0.94) and tightens
            the threshold for <Hi>56+</Hi> (whose model TPR was only 0.62).
            The mathematics are equal-opportunity correct; the policy
            implications deserve human review.
          </p>
        </Section>

        <Section eyebrow="07" title="Limitations">
          <ul className="my-4 list-none space-y-3">
            {[
              "Stratified 60K sample, not the full 307K — compute budget on commodity hardware.",
              "Single-attribute mitigation. Only age bucket was mitigated in this run; family status had ΔTPR 0.19 and remains future work.",
              "The hybrid underperformed LightGBM at this scale (0.7253 vs 0.7450 test AUROC). Honest ablation finding, not a hidden failure.",
              "No production traffic. Latency, drift, and adversarial-input behavior are all untested.",
            ].map((line) => (
              <li key={line} className="flex gap-3 text-cream-muted">
                <span className="mt-2 block h-px w-4 shrink-0 bg-cream/30" />
                <span className="leading-relaxed">{line}</span>
              </li>
            ))}
          </ul>
        </Section>

        <Section eyebrow="08" title="Credits">
          <p className="text-cream">
            Anupam Kumar · <Hi>Anish Kishore (primary contributor)</Hi> ·
            Swaraj Thakur
          </p>
          <p className="mt-2 text-sm text-cream-muted">
            BIT Mesra. Originally a Grab AI National Hackathon 2025
            semi-finalist project; rebuilt to a public-data benchmark for
            honest performance evaluation.
          </p>
          <div className="mt-8">
            <Link
              href="https://github.com/heynintendo/NovaScore-Credit-Scoring-System"
              className="group inline-flex items-center gap-3 rounded-full border border-cream/15 px-5 py-2.5 text-sm text-cream-muted transition-all hover:border-gold/50 hover:text-cream"
            >
              <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 0a8 8 0 00-2.5 15.6c.4.1.5-.2.5-.4v-1.5c-2.2.5-2.7-1-2.7-1-.4-.9-.9-1.2-.9-1.2-.7-.5.1-.5.1-.5.8.1 1.2.8 1.2.8.7 1.2 1.9.9 2.4.7.1-.5.3-.9.5-1.1-1.8-.2-3.6-.9-3.6-3.9 0-.9.3-1.6.8-2.2-.1-.2-.4-1 .1-2.1 0 0 .7-.2 2.2.8a7.6 7.6 0 014 0c1.5-1 2.2-.8 2.2-.8.4 1.1.2 1.9.1 2.1.5.6.8 1.3.8 2.2 0 3.1-1.9 3.7-3.6 3.9.3.2.6.7.6 1.5v2.2c0 .2.1.5.6.4A8 8 0 008 0z" />
              </svg>
              Read the source · GitHub
            </Link>
          </div>
        </Section>
      </div>
    </article>
  );
}

function Hi({ children }: { children: React.ReactNode }) {
  return <span className="text-cream">{children}</span>;
}

function Section({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="mt-20 space-y-5 first-of-type:mt-16"
    >
      <div className="flex items-baseline gap-4 border-b border-white/[0.05] pb-3">
        <span className="font-mono text-[10px] tracking-[0.25em] text-gold">
          {eyebrow}
        </span>
        <h2 className="font-display text-[clamp(1.5rem,3vw,2rem)] font-medium leading-tight text-cream">
          {title}
        </h2>
      </div>
      <div className="space-y-4 text-[17px] leading-[1.75] text-cream-muted">
        {children}
      </div>
    </motion.section>
  );
}

function Results() {
  const rows = [
    { name: "LightGBM (tuned baseline)", val: "0.7593", test: "0.7450" },
    { name: "Hybrid FT-Transformer + TCN", val: "0.7364", test: "0.7253" },
    { name: "Ensemble (val-AUROC-weighted)", val: "0.7585", test: "0.7456", emph: true },
  ];
  return (
    <div className="my-6 overflow-hidden rounded-2xl border border-white/[0.05]">
      <table className="w-full text-left">
        <thead>
          <tr className="bg-white/[0.02]">
            <th className="p-4 font-mono text-[10px] uppercase tracking-widest text-cream-muted">
              Model
            </th>
            <th className="p-4 text-right font-mono text-[10px] uppercase tracking-widest text-cream-muted">
              Val
            </th>
            <th className="p-4 text-right font-mono text-[10px] uppercase tracking-widest text-cream-muted">
              Test
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className="border-t border-white/[0.04]">
              <td className="p-4 font-display text-base text-cream">
                {r.emph ? <span className="italic">{r.name}</span> : r.name}
              </td>
              <td className="p-4 text-right font-mono text-sm tabular-nums text-cream-muted">
                {r.val}
              </td>
              <td className={`p-4 text-right font-mono text-sm tabular-nums ${r.emph ? "text-gold" : "text-cream"}`}>
                {r.test}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FairnessTable() {
  const rows = [
    { age: "18-25", tprBefore: "0.940", tprAfter: "0.801", adj: "+46.1" },
    { age: "26-40", tprBefore: "0.874", tprAfter: "0.797", adj: "+20.5" },
    { age: "41-55", tprBefore: "0.748", tprAfter: "0.799", adj: "−12.3" },
    { age: "56+", tprBefore: "0.617", tprAfter: "0.801", adj: "−42.1" },
  ];
  return (
    <div className="my-6 overflow-hidden rounded-2xl border border-white/[0.05]">
      <table className="w-full text-left">
        <thead>
          <tr className="bg-white/[0.02]">
            {["Age", "TPR before", "TPR after", "Score Δ"].map((h) => (
              <th
                key={h}
                className="p-4 font-mono text-[10px] uppercase tracking-widest text-cream-muted"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.age} className="border-t border-white/[0.04]">
              <td className="p-4 font-display text-base italic text-cream">{r.age}</td>
              <td className="p-4 font-mono text-sm tabular-nums text-cream-muted">{r.tprBefore}</td>
              <td className="p-4 font-mono text-sm tabular-nums text-cream">{r.tprAfter}</td>
              <td className="p-4 font-mono text-sm tabular-nums text-gold">{r.adj}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Inline schematic of the data flow — pure SVG, no external dependency. */
function ArchitectureSketch() {
  return (
    <div className="my-6 overflow-hidden rounded-2xl border border-white/[0.05] bg-white/[0.015] p-8">
      <svg
        viewBox="0 0 760 320"
        className="w-full"
        xmlns="http://www.w3.org/2000/svg"
        aria-label="Architecture: tabular tower and sequence tower fuse into a head"
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(232,220,196,0.6)" />
          </marker>
        </defs>

        <g fontFamily="var(--font-mono)" fontSize="11" letterSpacing="0.1em" fill="rgba(232,220,196,0.55)" textAnchor="middle">
          <text x="100" y="40">APPLICATION + AGGS</text>
          <text x="100" y="280">BUREAU 60-MONTH STATUS</text>
        </g>

        {/* Tabular input box */}
        <rect x="40" y="55" width="120" height="40" rx="8" stroke="rgba(232,220,196,0.18)" fill="rgba(15,27,44,0.6)" />
        <text x="100" y="80" textAnchor="middle" fill="#E8DCC4" fontFamily="var(--font-display)" fontSize="14" fontStyle="italic">80 features</text>

        {/* Sequence input box */}
        <rect x="40" y="225" width="120" height="40" rx="8" stroke="rgba(232,220,196,0.18)" fill="rgba(15,27,44,0.6)" />
        <text x="100" y="250" textAnchor="middle" fill="#E8DCC4" fontFamily="var(--font-display)" fontSize="14" fontStyle="italic">60 × 8 channels</text>

        {/* Arrows to towers */}
        <line x1="160" y1="75" x2="260" y2="105" stroke="rgba(232,220,196,0.5)" strokeWidth="1" markerEnd="url(#arrow)" />
        <line x1="160" y1="245" x2="260" y2="215" stroke="rgba(232,220,196,0.5)" strokeWidth="1" markerEnd="url(#arrow)" />

        {/* Towers */}
        <g>
          <rect x="260" y="80" width="180" height="50" rx="10" stroke="#C9A26F" strokeOpacity="0.6" fill="rgba(201,162,111,0.06)" />
          <text x="350" y="100" textAnchor="middle" fill="#E8DCC4" fontFamily="var(--font-display)" fontSize="15">FT-Transformer</text>
          <text x="350" y="118" textAnchor="middle" fill="rgba(232,220,196,0.55)" fontFamily="var(--font-mono)" fontSize="10">d_tab = 256 · 2 layers</text>
        </g>
        <g>
          <rect x="260" y="190" width="180" height="50" rx="10" stroke="#A8A8B3" strokeOpacity="0.6" fill="rgba(168,168,179,0.05)" />
          <text x="350" y="210" textAnchor="middle" fill="#E8DCC4" fontFamily="var(--font-display)" fontSize="15">TCN</text>
          <text x="350" y="228" textAnchor="middle" fill="rgba(232,220,196,0.55)" fontFamily="var(--font-mono)" fontSize="10">d_seq = 128 · 3 dilated blocks</text>
        </g>

        {/* Arrows to fusion */}
        <line x1="440" y1="105" x2="540" y2="150" stroke="rgba(232,220,196,0.5)" strokeWidth="1" markerEnd="url(#arrow)" />
        <line x1="440" y1="215" x2="540" y2="170" stroke="rgba(232,220,196,0.5)" strokeWidth="1" markerEnd="url(#arrow)" />

        {/* Fusion + head */}
        <rect x="540" y="135" width="170" height="50" rx="10" stroke="#D4B47A" strokeOpacity="0.7" fill="rgba(212,180,122,0.07)" />
        <text x="625" y="155" textAnchor="middle" fill="#E8DCC4" fontFamily="var(--font-display)" fontSize="15">Fusion MLP</text>
        <text x="625" y="173" textAnchor="middle" fill="rgba(232,220,196,0.55)" fontFamily="var(--font-mono)" fontSize="10">→ logit → score</text>

        {/* LGB parallel path - dashed */}
        <g>
          <text x="100" y="160" textAnchor="middle" fill="rgba(232,220,196,0.55)" fontFamily="var(--font-mono)" fontSize="10" letterSpacing="0.1em">LIGHTGBM (PARALLEL)</text>
          <line x1="160" y1="160" x2="540" y2="160" stroke="rgba(201,162,111,0.4)" strokeWidth="1" strokeDasharray="2 3" />
        </g>
      </svg>
    </div>
  );
}
