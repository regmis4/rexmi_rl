# REXMI YouTube Series — Production Bible

> **Series:** RL for Wheeled Quadrupeds — From Skills to Mission Autonomy  
> **Episodes:** 4  
> **Audience:** Beginners + interested engineers, robotics founders, technical investors  
> **Complexity:** Medium — concepts and systems, not reward-weight tutorials  
> **Primary CTA:** Personal brand as a serious RL / autonomy engineer for wheeled quadrupeds; soft investor interest in V4  
> **Last updated:** 2026-08-02  
> **Companion docs:** `project_roadmap.md`, `rl_setup.md`, `nav_layer.md`, `autonomous_nav_plan.md`, `lunar_crater_terrain_research.md`

---

## 1. Series Positioning

### One-sentence pitch

A four-part series on how to build a **multi-skill RL locomotion library and hierarchical autonomy stack** for a **wheeled hybrid quadruped**, using real project work on Unitree Go2W toward autonomous lunar-crater missions.

### What this is / is not

| This series **is** | This series **is not** |
|--------------------|------------------------|
| System design for RL robots | A click-by-click Isaac Lab tutorial |
| Lessons from real failures | A highlight reel that hides cliffs |
| Medium-depth RL + nav topics | Hyperparameter / reward-weight deep dives |
| Investor-credible engineering | Hype without capability envelopes |
| Path from now → mission profiles + SLAM | A claim that everything is already done |

### Brand lines (reuse 2–3 per video)

1. **“Visual gates beat metrics.”**
2. **“Specialize skills; compose systems.”**
3. **“The policy will exploit you — design for it.”**
4. **“Autonomy is interfaces plus integrity gates.”**
5. **“Wheels for efficiency, legs for impossible geometry.”**

### Host impression to leave

- Understands RL **and** physical constraints  
- Practices freeze policies, eval envelopes, one-theme-per-train  
- Builds layered products (skills → nav → missions), not one-off demos  
- Credible person to fund for wheeled-quadruped autonomy  

---

## 2. Narrative Arc

```
V1  Why this robot + how RL locomotion works
 ↓
V2  Flat rolling → insane terrain (skills, curriculum, eval)
 ↓
V3  A policy → a robot that navigates (planners, switching, recovery)
 ↓
V4  Navigation → missions, SLAM, lunar path, fundable end-state
```

**Investor through-line:**  
Wheeled quadrupeds + modern RL + hierarchical autonomy = a practical path to high-impact extreme-terrain and lunar-adjacent robots — and REXMI already has working pieces of that stack.

---

## 3. Scope Firewall (what each video owns)

| Topic | V1 | V2 | V3 | V4 |
|-------|:--:|:--:|:--:|:--:|
| Why wheeled hybrid / Unitree | ★ | | | brief |
| Energy vs pure-legged | ★ | | | brief |
| MDP / PPO / Isaac scale | ★ | | | |
| Reward-gaming philosophy | ★ | ★ | | |
| Terrain curriculum & generation | | ★ | | |
| Multi-skill library & eval | | ★ | mention | |
| Slope-turn / pulse skill | | concept | gap | milestone |
| Local/global planners | | | ★ | |
| Policy switching / recovery | | | ★ | |
| Risk / costmap | | | ★ | |
| SLAM | | | brief | ★ |
| Mission profiles | | | brief | ★ |
| Lunar g / ISRU / Artemis | tease | terrain science | | ★ |
| Investor roadmap / ask | | | | ★ |

---

## 4. Production Rules (all episodes)

### Do
- Open on real footage or a concrete failure, not a definition
- Use one architecture diagram per video
- Name physical limits (wheel radius, friction/slope, OOD commands)
- Credit Unitree, Isaac Lab, RSL-RL, PPO lineage
- End with “what we proved” + tease next episode
- Keep reward discussion at **intent** level (track, stabilize, escape stuck, climb when terrain justifies it)

### Don’t
- Read hyperparameter tables on camera
- Claim steep pivot or full mission autonomy before it is visually gated
- Bash Boston Dynamics / competitors — contrast morphologies instead
- Mix full nav + full lunar roadmap into V1–V2
- Ship “TensorBoard looked great” stories without the visual counterexample

### Thumbnail formulas
1. Go2W + “16 DOF” + net glyph  
2. Crater wall + angle callout (e.g. Shackleton ~31°)  
3. Split: costmap | robot | policy name  
4. Mission chips: TRAVERSE / SURVEY / RIM + Moon  

### Length targets
| Video | Target | Hard max |
|-------|--------|----------|
| V1 | 12–16 min | 18 min |
| V2 | 15–18 min | 20 min |
| V3 | 15–18 min | 20 min |
| V4 | 14–17 min | 18 min |

---

## 5. Capability Snapshot (speak from this; keep current)

> Update this table when checkpoints change. Do not improvise numbers on camera.

| Skill | Checkpoint (ref) | Role | Status |
|-------|------------------|------|--------|
| Fast flat | `model_1499.pt` | Up to ~2 m/s flat transit | Production |
| Rough | `model_8996.pt` | Generalist; stairs/boxes/slopes ~≤23° | **Frozen** production |
| Rocky slope | `model_13994.pt` | Boulder slopes ~15–35°, up+down | Production |
| Turn (flat) | `model_10992.pt` (Turn-B) | Stop → reorient on flat | Ready |
| Slope-turn | `model_13345.pt` @ ~20° | Pivot-ish on moderate slope | Partial; steep pulse in progress |
| Nav stack | `navigate.py` + `nav/*` | Mission → plan → switch → recover | Working; steep reorient blocked on slope-turn |
| SLAM | `slam.py` ICP + LiDAR | Pose + map with gates | Integrated; path to KISS-ICP/ROS |

**Honest gaps (say out loud in V3/V4):**
- Reliable heading change on **25–35°** not solved yet  
- Continuous yaw on steep faces is the wrong skill; **HOLD→YAW→SETTLE** is the path  
- Full operator “profile → unattended mission” is the end-state, not the present  

---

# VIDEO 1  
## Why Wheeled Quadrupeds + How RL Teaches Them to Move

**Draft titles**
- *Wheeled Quadrupeds & RL: The New Workhorse of Extreme Terrain*
- *Why Legs + Wheels Beat Pure Legs (and How RL Controls Them)*

**Role:** Hook + foundations. Establish host, platform, RL mental model.  
**Proof:** Flat / early full-DOF locomotion story; clear MDP picture.  
**CTA:** Subscribe — next video is where terrain gets insane.

### Learning outcomes
Viewer can explain:
1. Why a wheeled hybrid exists  
2. What a locomotion policy consumes and emits  
3. Why massively parallel sim changed the game  
4. Why reward gaming is normal and must be designed against  

### Run of show

| T+ | Beat | On-camera point | B-roll / visual |
|----|------|-----------------|-----------------|
| 0:00 | Cold open | “Not teleop — learned 16-DOF control on hard terrain.” | Best crater/rocky clip (even if from later work) |
| 0:45 | Who / series map | 4 videos: skills → terrain → nav → missions | Simple 4-box series graphic |
| 1:30 | Platform | Unitree Go2W: 12 leg DOF + 4 wheels | URDF/USD, real Unitree footage if licensed |
| 3:00 | Why hybrid | Energy & speed on rollable ground; legs when geometry breaks wheels | Diagram: roll vs step-over |
| 5:00 | Wheel physics teaser | ~5 cm wheel radius → pure roll limit; legs extend the envelope | Simple radius vs step sketch |
| 6:00 | RL loop | Policy, obs, action, reward, env | Loop diagram |
| 7:30 | MDP in plain language | State, action, transition=physics, reward, discount | Table graphic |
| 9:00 | What the robot sees/does | Vel, IMU/gravity, joints, cmd → 16 actions | Obs/action bars |
| 10:00 | Actuator interface | Soft PD legs, velocity-mode wheels | τ = PD sketch (light) |
| 11:00 | Why Isaac-scale sim | Thousands of envs on one GPU | 4096 robots mental image / play multi-env |
| 12:00 | PPO one level up | Roll out → advantage → clipped update; stable locomotion default | Tiny PPO card + paper cite |
| 13:00 | Phase story | Wheels-only → thighs/CG lean → full 16-DOF | Three short clips |
| 14:00 | First exploits | Spider-walk, leg splay; “policy exploits loopholes” | Blooper + fix label |
| 15:00 | Domain rand (light) | Mass, pushes, obs noise = distributional training | Event list graphic |
| 15:45 | Close + tease V2 | Flat is table stakes; cliffs and slopes are the craft | Stairs cliff teaser frame |

### Talking points (expanded)

**Wheeled hybrid value**
- Pure legged: extraordinary terrain, continuous foot work → energy cost on easy ground  
- Pure wheeled: efficient, fails on discrete obstacles and steep broken slopes  
- Hybrid: roll when you can, reconfigure legs when you must — **industrial sweet spot**  
- Unitree matters: real product SKU, not only a lab mechanism → deploy path  

**RL altitude**
- We are not “coding a gait.” We are specifying a task; optimization finds the gait  
- Command interface `(vx, vy, ωz)` is the contract between future nav and today’s muscle  
- Soft PD means the net outputs setpoints, not raw torques — safer structure for locomotion  

**Early lessons (from project history)**
- Phase 1: lock legs, prove wheels + stack  
- Phase 2: thigh pitch shifts CoM — “lean into the scooter”  
- Phase 3: full DOF unlocks mobility **and** gaming (contacts on shins, extreme splay)  
- Fix pattern: observe exploit → add structured penalty or termination → retrain  

### External references (name, don’t lecture)
- Schulman et al., PPO (2017)  
- Isaac Lab + RSL-RL  
- Unitree Go2W / unitree_ros descriptions  
- Optional: ETH RSL / ANYmal locomotion RL lineage  

### Demo commands (capture list)

```bash
conda activate env_isaacsim
cd /home/susan/rexmi_rl

# Flat baseline play
python scripts/play.py --task RexmiRl-Go2w-Velocity-Flat-Play-v0

# Fast flat (speed story)
python scripts/play.py --task RexmiRl-Go2w-Velocity-FastFlat-Play-v0
```

### V1 “what we proved”
- Hybrid morphology is a deliberate efficiency + mobility bet  
- RL locomotion is an MDP + scalable sim problem  
- Good engineering starts before terrain: interfaces, exploits, randomization  

---

# VIDEO 2  
## Terrain Intelligence: Curriculum, Multi-Skill Policies, Honest Evaluation

**Draft titles**
- *Teaching a Robot Insane Terrain (Without Lying to Yourself)*
- *From Flat to Crater Walls: Skills, Curriculum, and Eval*

**Role:** Core RL craft + real capability story.  
**Proof:** Rough + rocky-slope skills; eval cliffs; crater-bowl visuals.  
**CTA:** Next — the brain above the policies.

### Learning outcomes
1. Why skill libraries beat one mega-policy  
2. What terrain curriculum and procedural generation are for  
3. How height sensing changes the problem  
4. How to read a capability envelope  
5. Physical limits that no reward can repeal  

### Run of show

| T+ | Beat | On-camera point | B-roll / visual |
|----|------|-----------------|-----------------|
| 0:00 | Cold open | “It looked fine at 12 cm. At 15 cm it froze.” | Eval table + freeze clip |
| 0:45 | Thesis | Specialize skills; compose later | Skill library cards |
| 1:30 | Library tour | Fast flat / rough / rocky / turn | Four clips montage |
| 4:00 | Why split policies | Reward conflicts on unified steep+flat | Simple conflict diagram |
| 5:30 | Terrain generation | Stairs, boxes, rough, slopes, rocky pyramids | Tile grid visual |
| 7:00 | Curriculum | Rows = difficulty; advance with performance | terrain_levels sketch |
| 8:00 | Infra maturity | Curriculum state is part of the artifact | save/restore callout |
| 8:45 | Height scan | Local terrain image in the obs | Scan grid overlay |
| 10:00 | Reward intents | Track, live, don’t cheat contacts, escape stuck, climb when justified | Intent icons (no weights) |
| 12:00 | Exploit gallery | Bouncing climb farm; thigh salute; freeze attractor; metrics vs visual | Short bloopers |
| 13:30 | Stagnation lesson | Stuck with spinning wheels needs a progress pressure | Before/after stairs_down |
| 14:30 | Physics limits | Wheel radius; μ vs slope; CoM on faces | 2–3 clean diagrams |
| 16:00 | Eval culture | Fixed variants; tracking; survival; freeze checkpoints | `eval.py` output |
| 17:30 | Crater science hook | LOLA / Shackleton-class walls as target regime | Research fig |
| 18:15 | Close | Muscle without brain still isn’t a mission | Nav dashboard teaser |

### Talking points (expanded)

**Multi-skill library**
- Fast flat: efficiency and speed on easy ground (wheel action scale story at concept level)  
- Rough: generalist discrete obstacles + moderate slopes  
- Rocky slope: simultaneous steepness + boulders (crater-relevant)  
- Turn: reorient without requiring the forward specialist to be OOD at vx≈0  
- **Do not retrain frozen production** for curiosity — new run names, new experiments  

**Curriculum & generation**
- Procedural diversity beats hand-authored single tracks for robustness  
- Difficulty scheduling prevents “always die → no gradient”  
- Rocky slope tiles: slope **and** Gaussian boulders — matches bowl demo needs  
- Science-backed crater geometry (LOLA-informed) for investor-facing demos  

**Reward shaping (conceptual only)**
- Primary: follow velocity command  
- Safety: orientation, illegal contacts, joint limits  
- Progress: stagnation penalty when commanded but not moving  
- Climb: reward upward progress when terrain indicates obstacle — not on flat (anti-bounce)  
- Dead-zone penalties: free in normal ROM, costly only for pathological poses  

**Evaluation**
- Training terrain uses ranges → cannot say “fails at 15 cm” without fixed probes  
- `eval.py`: fixed command, fixed geometry, survival + tracking + distance  
- Deployment rule of thumb: high tracking **and** high survival define the envelope  
- **Visual gates beat TensorBoard** (preview slope-turn wiggle failure without full deep dive)

**Physics credibility**
- Wheel radius ~5 cm → wall-face contact on tall steps  
- Static hold roughly needs μ ≥ tan(θ); lunar g does not relax that inequality  
- Continuous spin on steep slopes fights support polygon — micro-reorient is the skill  

### Demo commands

```bash
# Rough production
python scripts/play.py --task RexmiRl-Go2w-Velocity-Rough-Play-v0

# Rocky slope
python scripts/play.py --task RexmiRl-Go2w-Velocity-RockySlope-Play-v0 \
  --load_run go2w_velocity_rocky_slope/2026-06-30_09-31-48 \
  --checkpoint model_13994.pt

# Crater bowl (headline terrain visual)
python scripts/play.py --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --load_run go2w_velocity_rocky_slope/2026-06-30_09-31-48 \
  --checkpoint model_13994.pt

# Eval examples
CKPT=logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt
python scripts/eval.py --checkpoint $CKPT --group stairs_down
python scripts/eval.py --checkpoint $CKPT --visual --terrain stairs_down_12cm

CKPT_ROCKY=logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt
python scripts/eval.py --checkpoint $CKPT_ROCKY --group rocky_slope_up
```

### V2 “what we proved”
- Terrain capability is engineered via skills, curriculum, and eval — not vibes  
- Physical limits and exploit dynamics are first-class  
- We can point to frozen checkpoints with known envelopes  

---

# VIDEO 3  
## Hierarchical Autonomy: Planners, Policy Switching, Recovery, Risk

**Draft titles**
- *RL Is the Muscle. This Is the Brain.*
- *Hierarchical Autonomy for Wheeled Quadrupeds*

**Role:** Systems engineering differentiator.  
**Proof:** `navigate.py`, dashboard, policy switches, recovery FSM.  
**CTA:** Next — missions, SLAM integrity, lunar end-state.

### Learning outcomes
1. Draw the autonomy layer cake  
2. Explain costmap risk at a high level  
3. Why hysteresis exists in policy switching  
4. Recovery as FSM + skills, not one giant net  
5. Why slope reorient is the critical open dependency  

### Run of show

| T+ | Beat | On-camera point | B-roll / visual |
|----|------|-----------------|-----------------|
| 0:00 | Cold open | Policy name flips on live dashboard | Dashboard + robot |
| 0:40 | Thesis | RL doesn’t replace planning; it makes actuators trainable | Layer cake diagram |
| 1:30 | Architecture | Mission → global → local → selector → RL | Animated stack |
| 3:00 | Sensing split | Height scan, forward cone, LiDAR map | Sensor FOV graphic |
| 4:30 | Noise vs truth | Policy gets noisy obs; planners get clean geometry | Two-path diagram |
| 5:30 | Global plan | A* on occupancy; lookahead waypoint | Path on costmap |
| 7:00 | Local plan | Heading candidates + speed scale + obstacle brake | 5-candidate sketch |
| 8:30 | Risk / cost | Unknown, steep, blocked — conservative costs | Cost legend |
| 10:00 | Policy selector | Terrain metrics → skill; asymmetric hysteresis | Decision tree |
| 12:00 | Command injection | Nav writes `(vx,vy,ωz)` into the same interface policies trained on | Tensor inject sketch |
| 13:00 | Recovery | Stuck → reverse → rotate → retry → replan | FSM diagram + clip |
| 14:30 | Turn override | Large heading error must not use OOD forward specialist | he>75° callout |
| 15:30 | Open gap (honest) | Steep stop-turn-go blocked on slope-turn skill; pulse FSM path | Hold vs wiggle clips |
| 17:00 | Engineering morals | Interfaces (`Localizer`), gates, one theme per train | Checklist graphic |
| 17:45 | Close | Nav is mid-game; missions + SLAM are endgame | Mission chips teaser |

### Talking points (expanded)

**Layer cake**
- Layer 1: RL skills (fast_flat, rough, rocky_slope, turn/slope-turn)  
- Layer 2: local speed/heading + selector + recovery  
- Layer 3: global path on map  
- Layer 3b/4: mission profiles (deep dive in V4)  
- Future: learned selector only after rule-based baseline exists  

**Why this architecture**
- Different skills have different obs dims / nets — selector loads multiple policies  
- Hysteresis: commit to safer skills fast; allow fast_flat only after sustained easy terrain  
- Recovery rotation must select a turn-capable policy — forward specialists at vx=0 stall  

**Planners**
- Map still small → A* is enough; replan on interval / deviation  
- Short height-scan horizon → few heading candidates beat heavy DWA  
- Forward scanner adds seconds of warning vs 1.5 m scan alone  

**Risk estimation**
- Costmap encodes traversability risk, not just occupancy binary  
- Unknown space penalized — autonomy should not assume freespace  
- Blocked cells force replan; steep cells expensive but sometimes passable  

**Slope-turn honesty**
- Flat turn: curriculum walk → arc → pivot; exploits catalogued and countered  
- Slope: hold often works before yaw does  
- Metrics can look healthy while robot only wiggles (Pulse20-v1 lesson)  
- Right skill: finite Δheading pulses, not endless ω tracking  
- Nav will emit the same pulse language the policy trains on  

### Demo commands

```bash
python scripts/navigate.py \
  --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Play-v0 \
  --ckpt_fast_flat logs/rsl_rl/go2w_velocity_fast_flat/2026-06-17_20-08-58/model_1499.pt \
  --ckpt_rough     logs/rsl_rl/go2w_velocity_rough/2026-06-14_20-03-41/model_8996.pt \
  --ckpt_rocky     logs/rsl_rl/go2w_velocity_rocky_slope/2026-06-30_09-31-48/model_13994.pt \
  --ckpt_turn      logs/rsl_rl/best_policies/go2w_turn_flat_v6_model_10992.pt \
  --mission traverse

# Optional headless capture
python scripts/navigate.py ... --mission survey --no_dashboard
```

### V3 “what we proved”
- A real hierarchical stack exists above RL  
- Switching, risk, and recovery are engineered behaviors  
- We know the remaining skill gap and how it blocks full obstacle loops  

---

# VIDEO 4  
## Mission Autonomy: SLAM, Profiles, and the Path to Lunar Ops

**Draft titles**
- *From Demo to Mission: SLAM, Profiles, and Lunar Reality*
- *The Full Stack: Autonomous Crater Missions*

**Role:** End-state vision + fundable milestones; grounded in current code.  
**Proof:** Mission modes, SLAM path, lunar research, roadmap discipline.  
**CTA:** Soft investor / collaborator interest; follow series.

### Learning outcomes
1. Mission profile = operator-facing product API  
2. SLAM with integrity gates vs “cheat pose”  
3. Why lunar sites (Shackleton et al.) matter  
4. Capital-efficient milestone ladder  
5. Why this host’s engineering culture de-risks funding  

### Run of show

| T+ | Beat | On-camera point | B-roll / visual |
|----|------|-----------------|-----------------|
| 0:00 | Cold open | “Operator picks a profile, not a gait.” | Mission UI mock / CLI missions |
| 0:45 | End-state shot | Autonomous profile on science-grade crater | Best composite trailer |
| 1:30 | Missions today | Traverse / survey / rim_circuit | Three path overlays |
| 3:30 | Product framing | Crater detection, nav, scan as profiles | Profile cards |
| 4:30 | SLAM story | LiDAR → voxels → ICP → gated pose | slam.py architecture |
| 6:30 | Integrity | Bootstrap, RMS reject, fallback odom | Gate diagram |
| 7:30 | Map as asset | Exploring builds the world model | Dashboard cloud growth |
| 8:30 | Close the loop | Hazard → stop → reorient → replan → resume | Full loop animation |
| 10:00 | Gap → milestone | Slope pulse turn unlocks steep detours | Roadmap node highlight |
| 11:00 | Lunar science | LOLA craters; Shackleton ~31°; Artemis/ISRU | Research charts |
| 12:30 | Lunar engineering | 1/6 g retrain; friction rand; energy metrics; recovery later | Phase icons |
| 14:00 | Industry frame | Unitree hybrids vs legs vs rovers | Comparison slide |
| 15:00 | Engineering culture | Freeze, eval, visual gates, delete failed exps, interfaces | Culture checklist |
| 16:00 | Soft ask | Milestones worth funding; contact/follow | Milestone ladder |
| 16:45 | Series close | Skills → systems → missions → product | Four-video recap |

### Talking points (expanded)

**Mission profiles (from `mission.py`)**
- **Traverse:** approach → rim → floor → exit — capability showcase  
- **Survey:** lawnmower on floor — prospecting / inspection analogue  
- **Rim circuit:** perimeter before committing to descent  
- Waypoints are crater-relative (rim/floor fractions) — scales to new geometry  

**SLAM & localization**
- Scripted ground-truth pose validates planners; it is not field autonomy  
- Project path: simulated Unitree L1-class LiDAR, voxel map, point-to-plane ICP  
- Gates: bootstrap window, RMS threshold, consecutive stable frames, odom fallback  
- Upgrade path without rewrite: KISS-ICP, ROS 2 pose topics, LIO-SAM-class fusion  
- Principle: **estimators fail; integrity logic makes systems survivable**  

**Lunar narrative (accurate, not sci-fi)**
- South pole PSRs and ice/ISRU are why agencies care  
- Shackleton-class walls ~30° are within the mobility story you’re building  
- Faustini / Nobile / de Gerlache give science and access-corridor color  
- Earth demos first; lunar-g fine-tune and regolith domain rand next  
- Full granular Chrono is a later infrastructure bet — say so  

**Fundable milestone ladder**
1. **M1** — Visually gated slope reorient (pulse) to ~25–35°  
2. **M2** — Full crater obstacle loop: detect → turn → A* detour → resume  
3. **M3** — Mission profiles + SLAM pose as default + operator dashboard  
4. **M4** — Lunar-g retrain + energy/risk metrics  
5. **M5** — Hardware path on Go2W-class platform / custom REXMI geometry  

**Soft ask language (suggested)**
> “If you care about extreme-terrain and lunar surface mobility, this stack is built as a product architecture — skills, nav, missions — with honest gaps and a clear milestone ladder. I’m looking for partners and capital to push slope reorient, full mission autonomy, and the hardware path.”

Avoid: overclaiming TRL, guaranteed timelines, or “solved Moon robot.”

### Demo commands

```bash
# Mission variants
python scripts/navigate.py ... --mission traverse
python scripts/navigate.py ... --mission survey
python scripts/navigate.py ... --mission rim_circuit

# Bowl record-style single robot (investor clean plate)
python scripts/play.py --task RexmiRl-Go2w-Crater-Bowl-RockySlope-Record-v0 \
  --load_run go2w_velocity_rocky_slope/2026-06-30_09-31-48 \
  --checkpoint model_13994.pt
```

### V4 “what we proved”
- The product is mission profiles on top of skills + nav + map integrity  
- Lunar relevance is grounded in real crater geometry and agency context  
- Funding buys a milestone ladder, not a mystery  

---

## 6. Cross-Series Shot List (capture once, cut many times)

### A. Hero mobility
| Shot | Source | Used in |
|------|--------|---------|
| Multi-robot crater bowl | `Crater-Bowl-RockySlope-Play` | V1 open, V2, V4 |
| Single-robot bowl record | `...-Record-v0` | V2, V4 |
| Fast flat sprint | FastFlat play | V1, V2 |
| Rough stairs/boxes | Rough play + eval visual | V2 |
| Rocky uphill/downhill | Rocky play / eval | V2 |
| Flat turn pivot | Turn-B play | V2, V3 |
| Slope hold on 25° | Slope-turn hold play | V3 gap |
| Slope wiggle failure | Pulse fail visual | V3 honesty |

### B. Autonomy
| Shot | Source | Used in |
|------|--------|---------|
| Full dashboard traverse | `navigate.py` traverse | V3, V4 |
| Policy switch moment | Dashboard status + log | V3 |
| Costmap + A* path | Dashboard right pane | V3, V4 |
| SLAM cloud growth | Dashboard 3D | V3 brief, V4 |
| Recovery reverse/rotate | Navigate stuck scenario | V3 |
| Survey lawnmower | `--mission survey` | V4 |
| Rim circuit | `--mission rim_circuit` | V4 |

### C. Graphics to produce
- Series 4-box map  
- Hybrid energy vs legs diagram  
- MDP loop  
- Layer cake autonomy  
- Skill library cards  
- Policy selector decision tree  
- Recovery FSM  
- SLAM integrity gates  
- Mission profile cards  
- Milestone ladder  
- Shackleton slope callout (cite LOLA/NASA via research doc)  

### D. Screen captures
- `eval.py` ASCII/CSV table  
- TensorBoard *only* as “metrics can lie” contrast, not as proof  
- Repo architecture tree (optional, 5s)  

---

## 7. Per-Video Checklist (pre-publish)

### Content
- [ ] Numbers match §5 capability snapshot  
- [ ] No reward-weight tables  
- [ ] At least one real failure + fix  
- [ ] Next episode teased  
- [ ] Brand line used ≥1×  

### Legal / credit
- [ ] Unitree name used respectfully  
- [ ] NASA/LOLA figures cited if shown  
- [ ] Third-party robot clips licensed or clearly fair-use short  

### Packaging
- [ ] Title + description + chapters  
- [ ] Thumbnail text ≤ 4 words dominant  
- [ ] Pins: link to series playlist + GitHub if public  

### Description blurb template

```
REXMI — RL + autonomy for wheeled quadrupeds (Unitree Go2W / Isaac Lab).

This episode: <one line>.

Series:
1) Foundations — wheeled hybrids + RL locomotion
2) Terrain skills — curriculum, multi-policy, eval
3) Nav stack — planners, switching, recovery
4) Missions + SLAM + lunar path

Not a hyperparameter tutorial — systems and lessons from a real stack.
```

---

## 8. Investor One-Pager (narration source for V4 close)

### Problem
Extreme-terrain and lunar surface ops need mobility beyond pure rovers and energy sense beyond pure legged continuous walking — plus software that turns low-level skills into **missions**.

### Solution thesis
1. **Morphology:** wheeled hybrid (efficiency + terrain)  
2. **Skills:** specialized RL policies with frozen envelopes  
3. **System:** hierarchical nav (map, plan, switch, recover)  
4. **Product API:** mission profiles (traverse, survey, rim, future scan/detect)  
5. **Integrity:** SLAM/localization gates, eval culture, honest gaps  

### Traction (technical)
- Production locomotion skills on flat, rough, and rocky slopes  
- Crater-bowl demo terrain informed by lunar south-pole geometry  
- Working nav stack with multi-policy switching and recovery  
- SLAM integration path with upgrade story to field estimators  
- Documented engineering process (roadmaps, eval, freeze policy)  

### Gaps (credibly stated)
- Steep-slope reorient skill in progress  
- Full unattended mission profiles on SLAM pose as default still maturing  
- Lunar-g and hardware deployment are roadmap, not complete  

### Use of funds (example framing)
- Close slope-turn → obstacle autonomy loop  
- Harden mission + SLAM dashboard for demos  
- Lunar-g retrain + energy/risk metrics  
- Go2W-class real-world bring-up  

### Why this team signal
- Physics-aware RL  
- Systems layering  
- Eval and freeze discipline  
- Willingness to kill failed experiments  

---

## 9. Optional 3-Video Compress (if needed)

If time forces 3 episodes:
1. **V1+V2 merge** light foundations + terrain skills (22 min max — risky)  
2. **Nav stack** (= current V3)  
3. **Missions + lunar** (= current V4)  

**Recommendation:** keep **4**. Foundations deserve airtime for non-RL investors.

---

## 10. Filming Order (efficient)

1. Capture **hero mobility** and **eval** clips (V1–V2)  
2. Capture **navigate + dashboard** long takes (V3–V4)  
3. Record **graphics** once  
4. Film **A-roll** V2 → V3 → V1 → V4 (V1 easiest once confidence is high; V4 last for sharp ask)  
5. Cut trailers from V2/V3 heroes for channel art  

---

## 11. File Map for Host Prep

| Deep dive while scripting | Doc |
|---------------------------|-----|
| MDP, phases, eval | `docs/rl_setup.md` |
| Roadmap / ROI | `docs/project_roadmap.md` |
| Nav modules | `docs/nav_layer.md` |
| Turn / slope-turn status | `docs/turn_policy_development.md`, `docs/slope_turn_policy_development.md` |
| Command semantics | `docs/turn_command_semantics.md` |
| Crater science | `docs/lunar_crater_terrain_research.md` |
| Demo runbook | `docs/lunar_crater_demo_run.md` |
| Autonomy gaps | `docs/autonomous_nav_plan.md` |

---

## 12. Episode Title Card (final candidates)

| # | Primary title | Subtitle |
|---|---------------|----------|
| 1 | Wheeled Quadrupeds + Reinforcement Learning | Why hybrids, and how policies actually walk |
| 2 | Terrain Skills for Insane Ground | Curriculum, multi-policy RL, and honest eval |
| 3 | The Autonomy Stack | Planners, policy switching, recovery, risk |
| 4 | Mission Autonomy Toward the Moon | SLAM, profiles, and the path to deployment |

---

*This document is the filming source of truth for the series. Update §5 when capabilities change. Do not contradict frozen checkpoint policy in `docs/rl_setup.md` / `docs/project_roadmap.md`.*
