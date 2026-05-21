"use client";

import { motion, useSpring, useTransform } from "motion/react";
import { useEffect } from "react";
import type { Tier } from "@/lib/utils";
import { TIER_COLOR } from "@/lib/utils";

interface Props {
  score: number;
  tier: Tier;
  /** Optional fairness-adjusted score for the secondary tick mark. */
  adjustedScore?: number;
}

const MIN = 300;
const MAX = 950;
const START_ANGLE = -210; // sweep start (top-left)
const END_ANGLE = 30;     // sweep end (top-right)

const SEGMENTS: { from: number; to: number; color: string; label: Tier }[] = [
  { from: 300, to: 600, color: TIER_COLOR.Bronze, label: "Bronze" },
  { from: 600, to: 700, color: TIER_COLOR.Silver, label: "Silver" },
  { from: 700, to: 800, color: TIER_COLOR.Gold, label: "Gold" },
  { from: 800, to: 950, color: TIER_COLOR.Platinum, label: "Platinum" },
];

const CENTER = { x: 200, y: 180 };
const RADIUS = 130;

function scoreToAngle(score: number): number {
  const t = Math.max(0, Math.min(1, (score - MIN) / (MAX - MIN)));
  return START_ANGLE + t * (END_ANGLE - START_ANGLE);
}

function polar(angleDeg: number, r: number): { x: number; y: number } {
  const a = (Math.PI / 180) * angleDeg;
  return { x: CENTER.x + r * Math.cos(a), y: CENTER.y + r * Math.sin(a) };
}

function arcPath(fromAngle: number, toAngle: number, r: number): string {
  const start = polar(fromAngle, r);
  const end = polar(toAngle, r);
  const large = Math.abs(toAngle - fromAngle) > 180 ? 1 : 0;
  // sweep flag = 1 (clockwise)
  return `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`;
}

export function ScoreGauge({ score, tier, adjustedScore }: Props) {
  const angle = useSpring(scoreToAngle(score), {
    stiffness: 120,
    damping: 18,
    mass: 0.8,
  });
  useEffect(() => {
    angle.set(scoreToAngle(score));
  }, [score, angle]);

  const displayScore = useSpring(score, { stiffness: 120, damping: 18 });
  useEffect(() => {
    displayScore.set(score);
  }, [score, displayScore]);
  const display = useTransform(displayScore, (v) => Math.round(v).toString());

  return (
    <div className="glass-warm relative overflow-hidden rounded-[28px] p-8 sm:p-10">
      {/* Subtle gold radial behind the gauge */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.18]"
        style={{
          background:
            "radial-gradient(circle at 50% 35%, rgba(201,162,111,0.45), transparent 65%)",
        }}
      />

      <div className="relative z-10 flex flex-col items-center">
        <p className="label-mono mb-6">Calibrated NovaScore</p>

        <svg
          viewBox="0 0 400 240"
          className="w-full max-w-[420px]"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Outer faint guide track */}
          <path
            d={arcPath(START_ANGLE, END_ANGLE, RADIUS + 14)}
            stroke="rgba(232,220,196,0.06)"
            strokeWidth={1}
            fill="none"
          />

          {/* Tier-colored segment arcs */}
          {SEGMENTS.map((seg) => {
            const a1 = scoreToAngle(seg.from);
            const a2 = scoreToAngle(seg.to);
            return (
              <path
                key={seg.label}
                d={arcPath(a1, a2, RADIUS)}
                stroke={seg.color}
                strokeWidth={10}
                strokeLinecap="butt"
                fill="none"
                opacity={tier === seg.label ? 1 : 0.32}
                style={{ transition: "opacity 0.4s ease" }}
              />
            );
          })}

          {/* Tier boundary tick marks */}
          {[600, 700, 800].map((s) => {
            const a = scoreToAngle(s);
            const outer = polar(a, RADIUS + 18);
            const inner = polar(a, RADIUS + 8);
            return (
              <line
                key={s}
                x1={inner.x}
                y1={inner.y}
                x2={outer.x}
                y2={outer.y}
                stroke="rgba(232,220,196,0.45)"
                strokeWidth={1}
              />
            );
          })}

          {/* End-point labels */}
          <text
            x={polar(START_ANGLE, RADIUS + 32).x}
            y={polar(START_ANGLE, RADIUS + 32).y}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="rgba(232,220,196,0.55)"
            fontFamily="var(--font-mono)"
            fontSize={10}
            letterSpacing={1.5}
          >
            300
          </text>
          <text
            x={polar(END_ANGLE, RADIUS + 32).x}
            y={polar(END_ANGLE, RADIUS + 32).y}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="rgba(232,220,196,0.55)"
            fontFamily="var(--font-mono)"
            fontSize={10}
            letterSpacing={1.5}
          >
            950
          </text>

          {/* Optional adjusted-score ghost tick */}
          {adjustedScore !== undefined && Math.abs(adjustedScore - score) > 1 && (
            <Tick angle={scoreToAngle(adjustedScore)} color="rgba(232,220,196,0.5)" />
          )}

          {/* Needle */}
          <motion.g
            style={{
              transformOrigin: `${CENTER.x}px ${CENTER.y}px`,
              rotate: angle,
            }}
            initial={false}
          >
            {/* The needle is positioned along angle 0 (East); the rotation
                handles aiming it. Length stops short of arc for elegance. */}
            <line
              x1={CENTER.x}
              y1={CENTER.y}
              x2={CENTER.x + RADIUS - 22}
              y2={CENTER.y}
              stroke={TIER_COLOR[tier]}
              strokeWidth={2.5}
              strokeLinecap="round"
            />
            <circle cx={CENTER.x} cy={CENTER.y} r={8} fill="#0A1628" stroke={TIER_COLOR[tier]} strokeWidth={2} />
            <circle cx={CENTER.x} cy={CENTER.y} r={3} fill={TIER_COLOR[tier]} />
          </motion.g>
        </svg>

        {/* Big number — Fraunces 96 */}
        <motion.div
          key="score-readout"
          className="-mt-2 font-display text-[clamp(4rem,9vw,6rem)] font-bold leading-none tabular-nums text-cream"
        >
          <motion.span>{display}</motion.span>
        </motion.div>

        {/* Tier label below number */}
        <p
          className="mt-3 font-display text-2xl italic tracking-tight"
          style={{ color: TIER_COLOR[tier] }}
        >
          {tier}
        </p>
      </div>
    </div>
  );
}

function Tick({ angle, color }: { angle: number; color: string }) {
  const outer = polar(angle, RADIUS - 14);
  const inner = polar(angle, RADIUS - 24);
  return (
    <line
      x1={inner.x}
      y1={inner.y}
      x2={outer.x}
      y2={outer.y}
      stroke={color}
      strokeWidth={1.5}
      strokeDasharray="2 2"
    />
  );
}
