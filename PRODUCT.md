# Product

## Register

product

## Users
Students, lab scientists, and engineers fitting a model to experimental data. They arrive with a CSV/XLSX (or values to key in by hand), pick a model (polynomial, inverse-power, or exponential), and need to (a) see the best-fit curve over their scatter and (b) read the same data as a straight line via a linearizing transform, ranked by R². The context is analytical and a little impatient: they want the fit and the equation, fast, and they trust the tool more when it looks like real measurement software.

## Product Purpose
A single-page regression + linearization workbench (Flask-served `index.html` / `styles.css`, Plotly + Handsontable + MathJax). It exists so curve fitting and the linearization technique taught in physics/lab courses can be done in the browser without a notebook. Success = a user loads data and gets a correct, legible fit + linearized view in seconds, and the equation/R² are unambiguous.

## Brand Personality
Precise, editorial, confident. Three words: **measured, kinetic, exacting.** It should feel like a well-set scientific journal that happens to compute — typographic authority, real figure apparatus (FIG. numbers, coordinates, equations set in math type), one hot signal accent. Not playful for its own sake; the motion is the instrument coming to life, not decoration.

## Anti-references
- Generic SaaS dashboards: blue/purple gradient hero backgrounds, glassmorphic floating cards, gradient orbs, animated sweep lines, button shimmer (this is the AI-default look the user explicitly rejected).
- Cream/sand/parchment "warm editorial" near-white backgrounds — the saturated AI warm-neutral.
- Hero-metric template, identical icon-card grids, tracked uppercase eyebrow above every section.

## Design Principles
- **The figure is the hero.** Data rendered as art carries the page; chrome stays quiet around it.
- **Typographic authority over decoration.** Hierarchy, scale, and a real display/mono/sans system do the expressive work — not effects.
- **Motion is the instrument waking up.** Every animation maps to something real (a curve being plotted, a fit resolving, a state changing) and degrades to an instant, complete state under `prefers-reduced-motion`.
- **One signal color.** A single hot accent marks fits, primary actions, and live state; everything else is ink and paper. Semantic red/green stay reserved for error/success.
- **The tool must stay a tool.** Editorial expression concentrates in masthead/hero/figure apparatus; the workbench itself stays legible, dense, and fast.

## Accessibility & Inclusion
Body text ≥4.5:1; large/graphical accent uses a darkened ink variant when it carries text. Full `prefers-reduced-motion` path (no entrance animation, content visible by default, never gated on a reveal). Status conveyed by text + shape, not color alone. Keyboard-reachable controls, visible focus rings.
