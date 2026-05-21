"use client";

import { motion } from "motion/react";
import type { Dispatch, SetStateAction } from "react";
import type { FamilyStatus, ScoreRequest } from "@/lib/types";
import { SAMPLE_APPLICANT } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  values: ScoreRequest;
  setValues: Dispatch<SetStateAction<ScoreRequest>>;
}

const FAMILY_OPTIONS: FamilyStatus[] = [
  "Married",
  "Single / not married",
  "Civil marriage",
  "Separated",
  "Widow",
];

export function FeatureInput({ values, setValues }: Props) {
  const u = <K extends keyof ScoreRequest>(k: K) => (v: ScoreRequest[K]) =>
    setValues((s) => ({ ...s, [k]: v }));

  return (
    <div className="space-y-12">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="label-mono">Applicant inputs</p>
          <h2 className="mt-2 font-display text-3xl font-medium leading-tight text-cream sm:text-4xl">
            Adjust signals.<br />
            <span className="italic text-cream/70">Watch the score answer.</span>
          </h2>
        </div>
        <button
          type="button"
          onClick={() => setValues(SAMPLE_APPLICANT)}
          className="group relative flex items-center gap-2 rounded-full border border-cream/15 px-4 py-2.5 text-xs uppercase tracking-widest text-cream-muted transition-all hover:border-gold/50 hover:text-cream"
        >
          <span className="block h-1.5 w-1.5 rounded-full bg-gold transition-transform group-hover:scale-125" />
          Use sample applicant
        </button>
      </header>

      <Section index="01" title="Demographics">
        <div className="grid gap-6 sm:grid-cols-2">
          <Slider
            label="Age"
            unit="years"
            min={18}
            max={70}
            value={values.age_years}
            onChange={(v) => u("age_years")(v)}
          />
          <SegmentedRadio
            label="Gender"
            options={[
              { v: "F", l: "Female" },
              { v: "M", l: "Male" },
            ]}
            value={values.gender}
            onChange={(v) => u("gender")(v as ScoreRequest["gender"])}
          />
          <Select
            label="Family status"
            value={values.family_status}
            options={FAMILY_OPTIONS}
            onChange={(v) => u("family_status")(v as FamilyStatus)}
          />
          <Slider
            label="Children"
            unit={values.num_children === 1 ? "child" : "children"}
            min={0}
            max={5}
            value={values.num_children}
            onChange={(v) => u("num_children")(v)}
          />
          <Toggle
            label="Has a car"
            value={values.has_car}
            onChange={(v) => u("has_car")(v)}
          />
        </div>
      </Section>

      <Section index="02" title="Financial">
        <div className="grid gap-6 sm:grid-cols-2">
          <CurrencyField
            label="Annual income"
            value={values.annual_income}
            min={20_000}
            max={500_000}
            step={5_000}
            onChange={(v) => u("annual_income")(v)}
          />
          <CurrencyField
            label="Loan amount requested"
            value={values.loan_amount}
            min={20_000}
            max={2_000_000}
            step={10_000}
            onChange={(v) => u("loan_amount")(v)}
          />
          <CurrencyField
            label="Annual annuity (repayment)"
            value={values.annuity}
            min={1_000}
            max={200_000}
            step={500}
            onChange={(v) => u("annuity")(v)}
          />
          <Slider
            label="Years employed"
            unit="yrs"
            min={0}
            max={40}
            value={values.years_employed}
            onChange={(v) => u("years_employed")(v)}
          />
        </div>
      </Section>

      <Section
        index="03"
        title="External credit signals"
        hint="Three independent bureau scores. Higher is stronger."
      >
        <div className="grid gap-6 sm:grid-cols-3">
          <Slider
            label="Bureau source 1"
            min={0}
            max={1}
            step={0.01}
            value={values.ext_source_1}
            onChange={(v) => u("ext_source_1")(v)}
            decimals={2}
          />
          <Slider
            label="Bureau source 2"
            min={0}
            max={1}
            step={0.01}
            value={values.ext_source_2}
            onChange={(v) => u("ext_source_2")(v)}
            decimals={2}
          />
          <Slider
            label="Bureau source 3"
            min={0}
            max={1}
            step={0.01}
            value={values.ext_source_3}
            onChange={(v) => u("ext_source_3")(v)}
            decimals={2}
          />
        </div>
      </Section>
    </div>
  );
}

/* ---------------- Form primitives — dark editorial ---------------- */

function Section({
  index,
  title,
  hint,
  children,
}: {
  index: string;
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="space-y-6"
    >
      <header className="flex items-baseline gap-4 border-b border-white/[0.05] pb-3">
        <span className="font-mono text-[10px] tracking-[0.2em] text-gold">{index}</span>
        <h3 className="font-display text-xl font-medium text-cream">{title}</h3>
        {hint && (
          <span className="ml-auto hidden text-xs italic text-cream-muted sm:block">
            {hint}
          </span>
        )}
      </header>
      {children}
    </motion.section>
  );
}

function Slider({
  label,
  unit,
  min,
  max,
  step = 1,
  value,
  onChange,
  decimals,
}: {
  label: string;
  unit?: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (v: number) => void;
  decimals?: number;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  const display = decimals !== undefined ? value.toFixed(decimals) : value.toString();
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <span className="label-mono">{label}</span>
        <span className="font-mono text-sm tabular-nums text-cream">
          {display} {unit && <span className="text-cream-muted">{unit}</span>}
        </span>
      </div>
      <div className="relative mt-3 h-7">
        {/* track */}
        <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-cream/15" />
        {/* filled */}
        <div
          className="absolute top-1/2 h-px -translate-y-1/2 bg-gold/80"
          style={{ width: `${pct}%` }}
        />
        {/* thumb */}
        <div
          className="pointer-events-none absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cream shadow-[0_0_0_4px_rgba(232,220,196,0.08)] ring-1 ring-cream/20 transition-transform"
          style={{ left: `${pct}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full cursor-pointer appearance-none bg-transparent opacity-0"
          aria-label={label}
        />
      </div>
    </label>
  );
}

function SegmentedRadio<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: Array<{ v: T; l: string }>;
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <fieldset>
      <legend className="label-mono">{label}</legend>
      <div className="mt-3 inline-flex rounded-full border border-cream/15 p-0.5">
        {options.map((opt) => (
          <button
            key={opt.v}
            type="button"
            onClick={() => onChange(opt.v)}
            aria-pressed={value === opt.v}
            className={cn(
              "rounded-full px-4 py-1.5 text-sm transition-colors",
              value === opt.v
                ? "bg-cream text-midnight"
                : "text-cream-muted hover:text-cream",
            )}
          >
            {opt.l}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="label-mono">{label}</span>
      <div className="relative mt-3">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none rounded-md border border-cream/15 bg-transparent px-4 py-2.5 pr-10 text-sm text-cream outline-none transition-colors focus:border-gold/60"
        >
          {options.map((o) => (
            <option key={o} value={o} className="bg-midnight text-cream">
              {o}
            </option>
          ))}
        </select>
        <svg
          aria-hidden
          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-cream-muted"
          width="14"
          height="14"
          viewBox="0 0 16 16"
          fill="none"
        >
          <path
            d="M4 6l4 4 4-4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    </label>
  );
}

function Toggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between">
      <span className="label-mono">{label}</span>
      <button
        type="button"
        onClick={() => onChange(!value)}
        role="switch"
        aria-checked={value}
        className={cn(
          "relative h-6 w-11 rounded-full border transition-colors",
          value ? "border-gold/50 bg-gold/25" : "border-cream/15 bg-cream/5",
        )}
      >
        <span
          className={cn(
            "absolute top-1/2 h-4 w-4 -translate-y-1/2 rounded-full transition-all",
            value ? "left-[calc(100%-1.25rem)] bg-gold" : "left-1 bg-cream/40",
          )}
        />
      </button>
    </label>
  );
}

function CurrencyField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <span className="label-mono">{label}</span>
      <div className="relative mt-3">
        <span className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-sm text-cream-muted">
          $
        </span>
        <input
          type="number"
          inputMode="numeric"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          className="w-full rounded-md border border-cream/15 bg-transparent py-2.5 pl-8 pr-4 font-mono text-sm tabular-nums text-cream outline-none transition-colors focus:border-gold/60"
        />
      </div>
    </label>
  );
}
