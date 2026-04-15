# Brainfast RC1 Cut Checklist

## Clean-room machine

1. Clone the repo at the candidate commit.
2. Create a fresh venv.
3. Run:
   ```powershell
   pip install -e ".[full,desktop,dev]"
   python project/scripts/check_env.py --config project/configs/run_config.template.json
   python -m pytest project/tests -q
   ```
4. Save:
   - `python -m pip freeze > release_candidate_requirements.txt`
   - `python -VV > python_runtime.txt`
   - `git rev-parse HEAD > git_commit.txt`
