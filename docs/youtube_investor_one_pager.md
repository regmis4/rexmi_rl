# REXMI — Investor One-Pager (Video Series Companion)

> **Use:** V4 close, deck appendix, or leave-behind.  
> **Tone:** Technical founder credibility. No TRL theater.  
> **Date:** 2026-08-02  
> **Detail:** `docs/youtube_series_plan.md`, `docs/project_roadmap.md`

---

## One-liner

**REXMI builds multi-skill reinforcement-learning locomotion and hierarchical mission autonomy for wheeled hybrid quadrupeds** — efficient like a rover where the ground allows, agile like a legged system where it does not — aimed at extreme-terrain and lunar-adjacent operations.

---

## Problem

| Gap | Why it hurts |
|-----|----------------|
| Pure rovers | Efficient on benign ground; limited on discrete obstacles and steep broken slopes |
| Pure legged | Extraordinary mobility; continuous stepping energy cost on long easy transits |
| “Train one policy” demos | Do not become operator products (missions, maps, recovery, integrity) |
| Lunar / PSR interest | Needs credible mobility **and** software that executes **profiles**, not joysticks |

Agencies and industry care about south-pole access, ISRU scouting, and unattended traverse. The missing piece is often the **full stack**: skills + navigation + mission API + localization integrity.

---

## Solution thesis

1. **Morphology** — Wheeled hybrid (Unitree Go2W class): roll for energy/speed; legs for geometry beyond wheel radius.  
2. **Skill library** — Specialized RL policies with **frozen** production checkpoints and measured envelopes.  
3. **Hierarchical nav** — Map → global plan → local traversability → policy selector → recovery FSM.  
4. **Mission profiles** — Operator selects *traverse / survey / rim recon* (path to detect & scan), not gaits.  
5. **Integrity** — SLAM/localization gates, eval culture, honest open gaps, upgradeable interfaces.

---

## What exists today (technical traction)

| Layer | Status | Evidence |
|-------|--------|----------|
| Fast flat locomotion | Production | ~2 m/s-class flat transit skill |
| Rough terrain generalist | **Frozen** production | Stairs/boxes/moderate slopes; eval harness |
| Rocky slope + boulders | Production | ~15–35° regime; crater-bowl demos |
| Flat reorient (turn) | Ready | Stop → turn for reroutes on flat |
| Slope reorient | **In progress** | ~20° turn asset; steep pulse skill active R&D |
| Nav stack | Working | Multi-policy switch, A*, costmap, recovery, dashboard |
| SLAM path | Integrated | LiDAR → voxel ICP, bootstrap/RMS gates, fallback |
| Mission generator | Working | Crater-relative traverse / survey / rim_circuit |
| Terrain science | Done (research) | LOLA-informed crater geometry; Shackleton-class ~30° walls |

**Engineering culture (de-risks execution):** checkpoint freeze policy, fixed-difficulty eval sweeps, curriculum save/restore, visual acceptance over vanity metrics, failed experiments deleted rather than left to contaminate configs.

---

## Honest gaps (we lead with these)

- Reliable **heading change on 25–35°** slopes not yet visually gated  
- Full unattended **profile → SLAM pose default → complete obstacle loop** still maturing  
- **Lunar gravity** retrain, regolith domain randomization, and **hardware** bring-up are roadmap  

Claiming otherwise would destroy the brand this series builds.

---

## Why wheeled hybrids (investor intuition)

- **Energy:** rolling beats continuous foot-lift on long easy segments  
- **Speed:** dedicated flat skill for transit between hard patches  
- **Terrain:** legs + body pitch extend capability past pure wheel geometry (~5 cm radius class limits)  
- **Industry timing:** Unitree-class platforms make the morphology purchasable, not only publishable  

---

## Product shape

```
Operator: select profile (traverse | survey | rim | future scan/detect)
    → Mission waypoints (crater-relative)
    → SLAM-localized nav (map + A* + local risk)
    → Policy library (flat | rough | rocky | turn)
    → Recovery when stuck
    → Dashboard: map, path, active skill, mission state
```

**End-state demo worth funding toward:**  
Operator selects *survey* or *traverse* on a science-grade crater; robot executes under gated localization with multi-skill mobility and recovery — no joystick.

---

## Milestone ladder (capital-efficient)

| ID | Milestone | Unlocks |
|----|-----------|---------|
| **M1** | Visually gated slope reorient (HOLD→YAW→SETTLE) toward 25–35° | Steep stop–turn–go |
| **M2** | Full crater obstacle loop (detect → turn → replan → resume) | True autonomous detour |
| **M3** | Missions + SLAM-as-default + demo-grade dashboard | Investor-facing autonomy product |
| **M4** | Lunar-g fine-tune + energy/risk metrics | Mission-planning credibility |
| **M5** | Go2W-class hardware path / custom REXMI geometry | Real-world TRL climb |

---

## Use of funds (example framing)

- Close **M1–M2** (highest demo ROI per `project_roadmap` prioritization)  
- Harden **mission + SLAM + dashboard** for repeatable demos  
- **Lunar-g** retrain campaign + friction randomization  
- **Hardware** bring-up budget (platform, compute, field sensing)  

---

## Competitive frame (non-hostile)

| Approach | Strength | Limitation |
|----------|----------|------------|
| Pure legged (e.g. classic quads) | Peak mobility | Energy on long easy ground |
| Pure rover | Heritage, efficiency | Discrete obstacles / agility |
| Single-policy RL demos | Viral clips | Weak product architecture |
| **REXMI bet** | Hybrid + skill library + missions | Must finish steep reorient + field path |

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Steep turn harder than expected | Pulse skill aligned with nav; visual gates; hold already demonstrated steeper than continuous yaw |
| Sim-to-real gap | Domain rand path; Unitree-class real platform; integrity-gated localization |
| Scope creep (Chrono granular, jumping) | Explicitly deferred; ROI-ordered roadmap |
| One-person bus factor | Documented stack (`docs/*`), frozen artifacts, clear interfaces |

---

## Ask

Partners and capital to execute **M1→M5** with emphasis on **slope reorient, full mission autonomy, and hardware path** — backing a team that already ships layered software, measures capability envelopes, and states gaps in public technical language.

**Contact / follow-through:** attach to V4 CTA and personal channels (fill in when publishing).

---

## Series as GTM

Four medium-depth videos educate the market and underwrite founder credibility:

1. Foundations — hybrid + RL locomotion  
2. Terrain skills — curriculum, multi-policy, eval  
3. Autonomy stack — planners, switching, recovery, risk  
4. Missions + SLAM + lunar path + ask  

*Skills make motion. Systems make a robot. Missions make a product.*
