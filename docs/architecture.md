1of1-nba-predictions/  # Root repo name
├── .github/           # GitHub-specific configs
│   └── workflows/     # CI/CD pipelines
│       └── daily-pipeline.yml  # The GitHub Action YAML for running the data pipeline
├── nba-pipeline/          # Backend/data pipeline (Python-based)
│   ├── src/           # Main code
│   │   ├── __init__.py
│   │   ├── ingestion.py  # NBA data pulling (nba_api, injuries, etc.)
│   │   ├── features.py   # Feature engineering logic
│   │   ├── model.py      # Training/inference (LightGBM/XGBoost)
│   │   └── utils.py      # Helpers (e.g., Supabase connection)
│   ├── models/        # Stored model artifacts (e.g., pickled LightGBM files)
│   │   └── v1.0.model
│   ├── requirements.txt  # Python deps (nba_api, pandas, lightgbm, supabase, etc.)
│   ├── run_pipeline.py   # Entry point script for the full batch job
│   ├── config.py         # Env vars, constants (e.g., API endpoints)
│   └── tests/            # Unit tests (e.g., pytest for features/inference)
├── web/               # Next.js dashboard (Vercel-hosted)
│   ├── app/           # Next.js 14+ app router structure
│   │   ├── page.tsx   # Root page (today's games table)
│   │   ├── games/     # Dynamic routes/filters
│   │   │   └── [date].tsx
│   │   └── api/       # API routes if needed (e.g., proxy Supabase for auth)
│   │       └── predictions/route.ts
│   ├── components/    # Reusable UI (e.g., GameTable, AuthGate)
│   ├── lib/           # Utilities (e.g., Supabase client init)
│   ├── public/        # Static assets (logos, etc.)
│   ├── .env.local     # Local env (Supabase URL/anon key — gitignore this)
│   ├── next.config.js # Vercel/Next configs
│   ├── package.json   # Node deps (next, react, @supabase/supabase-js, etc.)
│   ├── tsconfig.json  # TypeScript setup
│   └── vercel.json    # Deployment configs (e.g., cron if you add it later)
├── desktop/           # Tauri app (Rust + web UI)
│   ├── src/           # Rust backend
│   │   ├── main.rs    # Entry point
│   │   └── kalshi.rs  # Kalshi API integration (signing, execution)
│   ├── src-tauri/     # Tauri configs (tauri.conf.json for builds, keychain storage)
│   ├── public/        # Web assets if needed
│   ├── index.html     # Minimal web UI (e.g., predictions viewer + execute button)
│   ├── package.json   # If using JS for UI (e.g., React/Vite integration)
│   ├── Cargo.toml     # Rust deps (reqwest for API calls, rsa for signing, etc.)
│   └── tauri.conf.json # App config (window size, permissions for keychain/OS)
├── shared/            # Cross-component reusables (optional but useful)
│   ├── schemas/       # DB schema definitions (SQL files for Supabase tables)
│   │   └── init.sql   # CREATE TABLE scripts for nba_games, predictions, etc.
│   └── types/         # Shared TypeScript types (e.g., Prediction interface for web/desktop)
│       └── index.ts
├── docs/              # Documentation
│   ├── architecture.md  # High-level overview (like the original spec)
│   └── setup.md         # How to run locally (pipeline, web, desktop)
├── .env.example       # Template for env vars (Supabase keys, etc. — no secrets here)
├── .gitignore         # Ignore node_modules, .env, build artifacts, etc.
├── LICENSE            # e.g., MIT
└── README.md          # Project overview, setup instructions, how to deploy
