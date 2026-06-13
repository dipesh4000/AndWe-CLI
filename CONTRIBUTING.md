# Contributing Guide

Thank you for considering contributing to **KuroCode**!

## Getting Started

1. **Clone the repository** and install the development dependencies:
   ```bash
   git clone <repo-url>
   cd KuroCode
   uv sync --dev   # or `pip install -r requirements.txt`
   ```
2. **Run tests** to ensure the environment is set up correctly:
   ```bash
   pytest
   ```

## Workflow

- Create a feature branch: `git checkout -b feature/your-feature`.
- Write code following the project's style (ruff, mypy). Run `ruff check .` and `mypy src` locally.
- Add tests under the `tests/` directory.
- Submit a pull request targeting `main`.

## Code Style

- **Formatting**: `ruff format` (auto‑fixes layout).
- **Linting**: `ruff check .` – fix any reported issues.
- **Static typing**: `mypy src` – keep the code base strictly typed.

## CI

The GitHub Actions workflow runs **ruff → mypy → pytest** on every push. Ensure your contributions pass locally before pushing.

## License

Distributed under the MIT License. See `LICENSE` for more information.
