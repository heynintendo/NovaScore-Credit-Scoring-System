import type { ScoreRequest, ScoreResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:7860";

export async function scoreApplicant(req: ScoreRequest): Promise<ScoreResponse> {
  const r = await fetch(`${API_URL}/api/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    cache: "no-store",
  });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`Score API failed (${r.status}): ${body}`);
  }
  return (await r.json()) as ScoreResponse;
}

export const SAMPLE_APPLICANT: ScoreRequest = {
  age_years: 42,
  gender: "F",
  family_status: "Married",
  num_children: 1,
  has_car: true,
  annual_income: 175000,
  loan_amount: 550000,
  annuity: 27000,
  years_employed: 8,
  ext_source_1: 0.65,
  ext_source_2: 0.55,
  ext_source_3: 0.45,
};
