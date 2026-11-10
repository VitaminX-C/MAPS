<!-- FILE: .github/pull\_request\_template.md -->

\## What does this PR add?

<!-- One or two plain sentences. Example: "Replaces the placeholder in src/vision/preprocess.py with real blur and brightness checks." -->





\## File(s) changed

<!-- List every file, one per line. -->

\- 



\## How I tested it

<!-- Paste the commands you ran and what you saw. Example: `pytest` -> "7 passed". -->





\## Author checklist

\- \[ ] I read docs/CONTRACT.md and did not rename any file, function, key or label.

\- \[ ] Every file I changed is COMPLETE (no "..." and no "rest unchanged").

\- \[ ] The first line of every file is `# FILE: <exact path>`.

\- \[ ] Risky steps are inside try/except and return the safe default from the CONTRACT.

\- \[ ] I added or updated tests in tests/ and `pytest` passes.

\- \[ ] I added no libraries beyond the allowed list.

\- \[ ] I did not commit photos, databases, .env or .pt files (except models/best.pt).



\## Reviewer checklist

\- \[ ] Function names, file paths and data shapes match docs/CONTRACT.md exactly.

\- \[ ] The code never crashes on bad input (None image, empty text, missing model).

\- \[ ] Docstrings and plain-English comments are present.

\- \[ ] Tests are meaningful and run quickly with small fake data.

\- \[ ] I ran the app (`streamlit run app/main.py`) and it works.

