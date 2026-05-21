# NovaScore — frontend

A Next.js 15 + Tailwind v4 demo of the NovaScore credit scoring engine. Dark
editorial aesthetic; Fraunces display, Geist body, JetBrains Mono technical
labels; selective glassmorphism on the score gauge and navigation bar; motion
via the Motion library.

## Pages

- `/` — landing with the project tagline, dataset story, 3-L framework cards.
- `/score` — live calculator: a 12-field form on the left wired to a debounced
  call against the FastAPI backend, with the score gauge + decision card +
  fairness panel on the right.
- `/about` — long-form methodology, results table, fairness audit table,
  architecture sketch, limitations, credits.

## Run locally

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:7860 npm run dev
```

Open http://localhost:3000.

## Build

```bash
npm run build
npm start
```

## Deploy to Vercel

```bash
vercel --prod
```

Vercel auto-detects Next.js. Set `NEXT_PUBLIC_API_URL` to the deployed HF
Spaces API URL in the project's environment settings.

## Design tokens

Defined as CSS variables in `app/globals.css` under `@theme {}` (Tailwind v4
style). Reference them with `text-cream`, `bg-midnight`, `text-gold`, etc.

| Token | Value | Purpose |
|------|------|---------|
| `--color-midnight` | `#0A1628` | Base background |
| `--color-midnight-elevated` | `#0F1B2C` | Surfaces, cards |
| `--color-cream` | `#E8DCC4` | Primary text |
| `--color-cream-muted` | `rgba(232,220,196,0.65)` | Secondary text |
| `--color-gold` | `#C9A26F` | Accent, CTAs, Platinum tier |
| `--color-tier-gold` | `#D4B47A` | Gold tier band |
| `--color-tier-silver` | `#A8A8B3` | Silver tier band |
| `--color-tier-bronze` | `#8B6B4A` | Bronze tier band |
| `--color-tier-decline` | `#7A4848` | Error states |

## Typography

- **Display** (h1/h2/h3, hero, score number): Fraunces variable, optical sizing on
- **Body**: Geist Sans
- **Mono accents** (labels, technical numbers): JetBrains Mono

All loaded via `next/font/google` in `app/layout.tsx`.
