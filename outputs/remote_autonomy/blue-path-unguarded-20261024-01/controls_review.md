# Controls review — user-authorized unguarded simulation

**8.2/10: approve one unchanged-policy seed-42 simulation experiment after the executor's final tests pass.** No remaining concrete command-ownership or completion blocker found. This review accepts the user's explicit removal of motion guardrails; it does not require restoring terrain, footprint, tilt, speed, cross-track or staging vetoes.

Reviewed `experimental_path_control.py`, `--ascent_unguarded` integration in `navigate.py`/`checkpoint_runner.py`, inherited saved-route loading, and the new tests. No code edits, tests or simulation were run by this reviewer.

- The experimental tick bypasses inherited physical admission. STOP/manual ownership and invalid-observation zero output remain distinct from autonomous motion; the autonomous experiment retains its 900-second budget.
- Recovery owns exactly one turn, push or reverse action. Measured angular response, route/cross-track score, push duration and reverse displacement/time control transitions. Terrain values rank headings without rejecting them.
- Completion requires the actual selected endpoint, contiguous 0.6-second settling, low translation/yaw rate and measured upright state. Map gentleness is intentionally not required.
- Interrupted recovery actions are censored on STOP/manual, budget expiration, endpoint settling and finalization. Invalid observations retain the last valid evidence. Censored pushes are excluded from failed-push preference memory.
- Summary fields now report recovery actions/history and final/furthest route progress. Automatic climb-failure classification is explicitly unavailable, rather than falsely reporting zero physical failures. Detailed recovery history is capped at 64 entries in memory; the event stream preserves action records.
- Saved route/map hashes, terrain identity, exact input copy and executor source provenance remain recorded.

Validation evidence: executor reported 169 focused tests passing before the final censor-memory assertions; its final rerun is required before launch. Tests inspected cover removed vetoes, persistent turns, push/reverse response, productive release, STOP/manual/invalid data/budget/completion, and action censoring.

Interpretation limits: successful angular progress renews the response clock, so the eight-second plateau is not a strict positional-progress timeout during productive turns. Failed-push memory is a local measured-progress preference, not a validated physical-failure classifier. Approval is for the explicitly authorized simulation and learning objective, not hardware operation, safety qualification or predicted ascent success. Preserve the first run and do not patch live or retry automatically.
