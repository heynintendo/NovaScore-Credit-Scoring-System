"use client";

import Link from "next/link";
import { motion } from "motion/react";

export default function LandingPage() {
  return (
    <>
      {/* Hero — asymmetric, half-bleed, no center alignment except where deliberate. */}
      <section className="relative overflow-hidden pt-40 pb-32 lg:pt-48 lg:pb-44">
        <Ornament />
        <div className="relative mx-auto grid max-w-7xl grid-cols-12 gap-x-6 px-6 lg:px-10">
          {/* Eyebrow label */}
          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="col-span-12 mb-10 lg:col-span-6 lg:mb-16 label-mono"
          >
            Equitable Credit Scoring · v0.2 · Home Credit Default Risk
          </motion.p>

          {/* H1 stack */}
          <div className="col-span-12 lg:col-span-9">
            <motion.h1
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.65, ease: [0.16, 1, 0.3, 1], delay: 0.06 }}
              className="font-display text-[clamp(3.5rem,9vw,7rem)] font-bold leading-[0.95] text-cream"
            >
              Nova<span className="text-gold">Score</span>
              <span className="block font-display text-[clamp(1.25rem,2.4vw,2rem)] font-normal italic leading-snug text-cream/85 mt-6 max-w-[40ch] text-balance">
                Respecting those who carry trust, with credit that carries them.
              </span>
            </motion.h1>
          </div>

          {/* Subtext — asymmetric, sits to the right of hero, aligns to bottom */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, ease: [0.16, 1, 0.3, 1], delay: 0.18 }}
            className="col-span-12 mt-12 max-w-prose text-cream-muted lg:col-span-7 lg:col-start-1"
          >
            <p className="text-lg leading-relaxed text-pretty">
              A credit scoring engine for the people whose creditworthiness is
              invisible to traditional systems. Built on the Home Credit Default
              Risk public dataset — <span className="text-cream">307,511 real anonymized loan applications</span> —
              with calibrated <span className="text-cream">0.75 test AUROC</span> and a
              <span className="text-cream"> 99% reduction</span> in age-group disparity through fair threshold optimization.
            </p>
          </motion.div>

          {/* CTA + secondary link */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
            className="col-span-12 mt-10 flex flex-wrap items-center gap-4 lg:col-span-7"
          >
            <Link
              href="/score"
              className="group relative inline-flex items-center gap-3 rounded-full bg-gold px-7 py-3.5 text-sm font-medium text-midnight transition-all duration-300 hover:scale-[1.02] hover:shadow-[0_0_40px_-8px_rgba(201,162,111,0.55)]"
            >
              <span>Try the calculator</span>
              <ArrowOut />
            </Link>
            <Link
              href="/about"
              className="text-sm text-cream-muted underline decoration-cream/30 underline-offset-4 transition-colors hover:text-cream hover:decoration-gold"
            >
              Read the methodology
            </Link>
          </motion.div>

          {/* Right-side meta strip on desktop — score range etched into the page */}
          <motion.aside
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.9, delay: 0.5 }}
            className="hidden lg:col-span-3 lg:col-start-10 lg:row-start-2 lg:flex lg:flex-col lg:items-end lg:gap-6 lg:self-end lg:pb-2"
          >
            <ScoreScale />
          </motion.aside>
        </div>
      </section>

      {/* 3-L framework — three DIFFERENT card layouts, asymmetric placement */}
      <ThreeL />

      {/* Closing pull-quote */}
      <section className="relative pb-32 pt-24 lg:pt-32">
        <div className="mx-auto max-w-5xl px-6 lg:px-10">
          <p className="label-mono mb-8">A note on intent</p>
          <p className="max-w-4xl font-display text-[clamp(1.5rem,3vw,2.4rem)] font-normal leading-[1.2] tracking-tight text-cream/90 text-balance">
            Credit is a story the system tells about you. NovaScore changes who gets
            to write that story — and audits the writing for harm before it lands
            on a decision.
          </p>
        </div>
      </section>
    </>
  );
}

/** Gold-tinted ambient gradient orb + faint compass cross. Pure decoration. */
function Ornament() {
  return (
    <>
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/2 h-[640px] w-[640px] -translate-x-1/2 rounded-full opacity-[0.18] blur-3xl"
        style={{
          background:
            "radial-gradient(circle, rgba(201,162,111,0.55) 0%, rgba(201,162,111,0) 60%)",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute left-[6%] top-[34%] h-[420px] w-[420px] rounded-full opacity-[0.10] blur-[100px]"
        style={{
          background: "radial-gradient(circle, #C9A26F 0%, transparent 70%)",
        }}
      />
      {/* A faint hairline grid behind everything — discoverable on hover */}
      <svg
        aria-hidden
        className="pointer-events-none absolute inset-0 h-full w-full opacity-[0.025]"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern id="g" width="80" height="80" patternUnits="userSpaceOnUse">
            <path d="M 80 0 L 0 0 0 80" fill="none" stroke="#E8DCC4" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#g)" />
      </svg>
    </>
  );
}

function ArrowOut() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      className="transition-transform duration-300 group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
    >
      <path
        d="M4 12L12 4M12 4H6M12 4V10"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Tiny score-range scale rendered as etched marks on the right of the hero. */
function ScoreScale() {
  return (
    <div className="flex w-full max-w-[140px] flex-col items-end gap-3">
      <p className="label-mono-faint">Score range</p>
      <div className="relative h-44 w-full">
        <div className="absolute right-0 top-0 h-full w-px bg-cream/15" />
        {[
          { v: 950, label: "Platinum" },
          { v: 800, label: "" },
          { v: 700, label: "Gold" },
          { v: 600, label: "Silver" },
          { v: 300, label: "Bronze" },
        ].map(({ v, label }) => {
          const pos = ((950 - v) / (950 - 300)) * 100;
          return (
            <div
              key={v}
              className="absolute right-0 flex items-center gap-2 text-cream-muted"
              style={{ top: `${pos}%` }}
            >
              <span className="font-mono text-[10px] tabular-nums tracking-wider">
                {v}
              </span>
              <span className="block h-px w-3 bg-cream/30" />
              {label && (
                <span className="font-display text-xs italic text-cream/80">
                  {label}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Three cards, three distinct layouts — stat callout / editorial / quote. */
function ThreeL() {
  return (
    <section className="relative pb-24 pt-16 lg:pb-32 lg:pt-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-10">
        <div className="mb-16 flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="label-mono mb-4">A 3-L framework</p>
            <h2 className="max-w-2xl font-display text-[clamp(2rem,4.5vw,3.25rem)] font-medium leading-[1.05] text-cream">
              Learn the signals. Lend with care. Loop through fairness.
            </h2>
          </div>
          <p className="max-w-md text-sm leading-relaxed text-cream-muted lg:text-right">
            Three loops in sequence, each independently auditable. The model
            doesn&apos;t see protected attributes; the system measures them after the
            fact and corrects what it can.
          </p>
        </div>

        <div className="grid grid-cols-12 gap-x-6 gap-y-10">
          {/* Card 1 — Learn: dominant stat */}
          <motion.article
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            className="relative col-span-12 overflow-hidden rounded-3xl border border-white/[0.05] bg-[#0F1B2C] p-10 lg:col-span-7 lg:row-span-2 lg:p-14"
          >
            <p className="label-mono mb-6">01 · Learn → Signals → Features</p>
            <div className="flex flex-col gap-6 sm:flex-row sm:items-end sm:gap-10">
              <div className="font-display text-[clamp(4.5rem,11vw,7.5rem)] font-bold leading-none text-gold">
                80
              </div>
              <div>
                <p className="font-display text-2xl leading-tight text-cream">
                  Features, distilled from 165
                </p>
                <p className="mt-3 max-w-prose text-cream-muted">
                  Application demographics, financial ratios, three external bureau
                  scores, plus per-applicant aggregates from bureau, previous
                  applications, and installments tables. SelectKBest narrows to the
                  most predictive 80; the FT-Transformer tokenises each one.
                </p>
              </div>
            </div>

            {/* A subtle decorative scribble in the corner. */}
            <svg
              aria-hidden
              className="absolute -right-6 -bottom-6 h-32 w-32 opacity-[0.06]"
              viewBox="0 0 100 100"
            >
              <circle
                cx="50"
                cy="50"
                r="48"
                fill="none"
                stroke="#E8DCC4"
                strokeWidth="0.5"
                strokeDasharray="2 3"
              />
              <circle cx="50" cy="50" r="20" fill="none" stroke="#C9A26F" strokeWidth="0.5" />
            </svg>
          </motion.article>

          {/* Card 2 — Lend: editorial paragraph with embedded mini-viz */}
          <motion.article
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
            className="col-span-12 rounded-3xl border border-white/[0.05] bg-[#0F1B2C] p-10 lg:col-span-5"
          >
            <p className="label-mono mb-6">02 · Lend → Model → Score</p>
            <p className="font-display text-2xl leading-snug text-cream">
              A LightGBM baseline reaches{" "}
              <span className="text-gold">0.745 test AUROC</span>. A hybrid
              FT-Transformer + TCN explores deep architectures, and a val-AUROC-weighted
              ensemble narrowly improves on both.
            </p>
            <MiniBars />
          </motion.article>

          {/* Card 3 — Loop: quote-style framing */}
          <motion.article
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
            className="col-span-12 rounded-3xl border border-white/[0.05] bg-[#0F1B2C] p-10 lg:col-span-5"
          >
            <p className="label-mono mb-6">03 · Loop → Fairness → Trust</p>
            <blockquote className="border-l-2 border-gold/60 pl-6">
              <p className="font-display text-2xl italic leading-snug text-cream/90 text-pretty">
                The fairness loop audits, doesn&apos;t hide. ΔTPR drops from{" "}
                <span className="not-italic text-gold">0.32</span> to{" "}
                <span className="not-italic text-gold">0.004</span> across age
                buckets via per-group threshold equalization.
              </p>
            </blockquote>
            <p className="mt-6 label-mono">
              99.6% reduction · target TPR = 0.80
            </p>
          </motion.article>
        </div>
      </div>
    </section>
  );
}

/** Mini horizontal bar chart for the Lend card — three models, AUROC scale. */
function MiniBars() {
  const items = [
    { label: "LightGBM", v: 0.7450, color: "#C9A26F" },
    { label: "Hybrid", v: 0.7253, color: "#A8A8B3" },
    { label: "Ensemble", v: 0.7456, color: "#D4B47A" },
  ];
  const min = 0.65;
  const max = 0.80;
  return (
    <div className="mt-8 space-y-3">
      {items.map(({ label, v, color }) => {
        const w = ((v - min) / (max - min)) * 100;
        return (
          <div key={label} className="space-y-1.5">
            <div className="flex items-baseline justify-between font-mono text-[11px] uppercase tracking-widest text-cream-muted">
              <span>{label}</span>
              <span className="text-cream">{v.toFixed(4)}</span>
            </div>
            <div className="relative h-2 w-full overflow-hidden rounded-full bg-cream/[0.05]">
              <motion.div
                initial={{ width: 0 }}
                whileInView={{ width: `${w}%` }}
                viewport={{ once: true }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
                style={{ background: color }}
                className="absolute inset-y-0 left-0 rounded-full"
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
