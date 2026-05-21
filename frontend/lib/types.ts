import type { Tier } from "./utils";

export type Gender = "F" | "M";
export type FamilyStatus =
  | "Married"
  | "Single / not married"
  | "Civil marriage"
  | "Separated"
  | "Widow";

export interface ScoreRequest {
  age_years: number;
  gender: Gender;
  family_status: FamilyStatus;
  num_children: number;
  has_car: boolean;
  annual_income: number;
  loan_amount: number;
  annuity: number;
  years_employed: number;
  ext_source_1: number;
  ext_source_2: number;
  ext_source_3: number;
}

export interface ScoreResponse {
  pd: number;
  novascore: number;
  decision_band: Tier;
  policy: string;
  fairness_adjusted_score: number;
  fairness_adjusted_band: Tier;
  fairness_adjustment_reason: string | null;
  age_bucket: "18-25" | "26-40" | "41-55" | "56+";
  raw_features_used: number;
}
