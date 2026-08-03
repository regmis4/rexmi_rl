# REXMI YouTube — On-Camera Cue Cards

> Print or keep on a second monitor. Full bible: `docs/youtube_series_plan.md`  
> Rule: **no reward weights on camera.** Update capability lines if checkpoints change.

---

## Series open (any episode)

> “This series is about building a real autonomy stack for a wheeled quadruped —  
> specialized RL skills, then navigation, then missions.  
> Not a hyperparameter tutorial. Systems and lessons from a working project.”

**Brand lines (pick one per episode)**  
- Visual gates beat metrics.  
- Specialize skills; compose systems.  
- The policy will exploit you — design for it.  
- Autonomy is interfaces plus integrity gates.  
- Wheels for efficiency, legs for impossible geometry.

---

# V1 — Foundations (~12–16 min)

## Cold open
Not teleop. A learned policy driving sixteen actuators — twelve leg joints, four wheels — on serious terrain.

## Map
Four videos: how RL moves the robot → terrain skills → nav brain → missions and the lunar path.

## Platform
Unitree Go2W: wheeled hybrid. Production-class hardware, not only a lab mechanism.

## Why hybrid
- Legs alone: amazing terrain, expensive on easy ground.  
- Wheels alone: efficient, die on steps and broken slopes.  
- Hybrid: roll when you can; reconfigure when geometry demands it.  
- Energy and speed on flats; legs extend the envelope when the five-centimeter wheel hits a wall-face.

## RL in one minute
Neural net: observations in, joint commands out.  
Physics is the transition. Reward is the task definition.  
We specify the contract — optimization finds the gait.

## What it sees / does
Sees: body motion, gravity direction, joints, velocity command — later a height scan.  
Does: wheel velocity targets plus leg pose offsets.  
Interface to the future brain is always roughly: forward, lateral, yaw rate.

## Why sim scale matters
Thousands of robots on one GPU. Locomotion RL became an engineering loop, not a weekend experiment. Isaac Lab, RSL-RL, PPO — standing on that stack.

## Early project story
1. Wheels only, legs locked — prove the pipeline.  
2. Unlock thighs — center of mass lean, like a scooter.  
3. Full sixteen DOF — mobility and cheating appear together.  
Spider-walk on shins. Extreme leg splay.  
Lesson: the policy will exploit any loophole. Good RL is adversarial design.

## Domain randomization (light)
Mass, pushes, sensor noise — train for a distribution, not one perfect sim.

## Close
Flat rolling is table stakes.  
Next: stairs, boulders, crater walls — and how we measure where the robot actually dies.

---

# V2 — Terrain skills (~15–18 min)

## Cold open
At twelve centimeters of stair it looked fine.  
At fifteen it froze — wheels spinning, no progress.  
That cliff taught more than a thousand blind reward tweaks.

## Thesis
One mega-policy that does everything fights itself.  
Specialize skills. Freeze what works. Compose later.

## Library (say clearly)
- **Fast flat** — sprint and efficient transit, about two meters per second class.  
- **Rough** — generalist obstacles and moderate slopes — production, frozen.  
- **Rocky slope** — steep faces with boulders, up and down — crater relevant.  
- **Turn** — stop and reorient so navigation can reroute.

## Curriculum and terrain
Procedural tiles: stairs, boxes, rough, slopes.  
Difficulty rows that advance with performance.  
Rocky pyramids with boulders — not toy flat grids.  
Curriculum state is part of the artifact — we save and restore it. That’s infrastructure maturity.

## Height scan
A small terrain image under the belly.  
The policy can react to geometry it was never hard-coded for.

## Reward intents only
- Follow the command.  
- Stay alive and legal — no shin-walking, no joint insanity.  
- If commanded forward and stuck — pressure to escape.  
- Climb when the terrain justifies it — not bounce on flat to farm a climb signal.  
Dead zones: free in normal motion, costly only for pathological poses.

## Exploits gallery (pick 2–3)
Bouncing to farm climb reward.  
Thigh salute to unload wheels.  
Freeze on slopes when survival pays more than yaw.  
Beautiful TensorBoard, robot only wiggles.  
**Visual gates beat metrics.**

## Physics
Wheel radius limits pure rolling.  
Friction versus slope angle — static hold is geometry and mu, and lunar gravity does not repeal tan theta.  
Continuous spinning on a steep face fights the support polygon.  
The right skill is often: hold, micro-yaw, settle — not endless spin rate tracking.

## Eval culture
Training mixes difficulties — you cannot quote a single failure height from training alone.  
Fixed probes: tracking ratio, survival, distance.  
Capability envelope: where tracking and survival are both high.  
Frozen checkpoints. New ideas get new experiment names.

## Close
Muscle without a brain is not a mission.  
Next: planners, policy switching, recovery.

---

# V3 — Autonomy stack (~15–18 min)

## Cold open
Watch the dashboard: costmap, path, and the active policy name changing — fast flat, rough, rocky slope — no joystick.

## Thesis
Reinforcement learning is the muscle.  
This layer is the brain.  
RL does not replace planning; it makes the actuator layer trainable and robust.

## Layer cake
Mission goals  
→ global path on a map  
→ local speed and heading  
→ policy selector  
→ RL skill at fifty hertz.

## Sensing
Height scan: short-range traversability.  
Forward cone: brake before you kiss the boulder.  
Lidar: build the world.  
Policy observations are noisy on purpose. Planners get clean geometry. Different jobs.

## Global and local
A star on a costmap — small maps, fast replan.  
Locally: a handful of heading candidates scored by clearance and slope — matched to a short sensor horizon.  
Heavy sampling planners are optional when you only see a body-length ahead.

## Risk
Cells are not only free or occupied.  
Unknown is expensive. Steep is expensive. Blocked is infinite.  
That is lightweight risk estimation for traversal.

## Policy switching
Terrain metrics pick the skill.  
Hysteresis: switch to safer policies quickly; only allow high-speed flat after the world stays easy.  
Large heading error forces a turn-capable skill — forward specialists at zero forward command are out of distribution and stall.

## Recovery
Stuck on speed or progress: reverse, rotate, retry, then replan.  
Often a state machine calling skills — not one giant “life policy.”

## Honest gap
Obstacle loops need stop — turn — go on slopes.  
Flat turn: in good shape.  
Steep turn: hold works earlier than yaw.  
We train pulse reorient — hold, yaw, settle — and we do not ship wiggle metrics.  
Until that skill is visually gated, steep detours are incomplete. That’s engineering, not failure theater.

## Close
Navigation is the mid-game.  
Next: mission profiles, SLAM integrity, and the path toward lunar ops.

---

# V4 — Missions, SLAM, lunar path (~14–17 min)

## Cold open
The operator should not pick a gait.  
They pick a mission profile — traverse, survey, rim recon, later detect and scan.

## Product API
- **Traverse** — enter, cross, exit — prove mobility.  
- **Survey** — lawnmower on the floor — inspection and prospecting analogue.  
- **Rim circuit** — perimeter before you commit to descent.  
Waypoints scale with crater geometry — mission code, not hard-coded demo coordinates only.

## SLAM
Scripted simulator pose validates planners. It is not field autonomy.  
Our path: lidar scans, voxel map, ICP, pose with bootstrap and error gates and fallback.  
Estimators fail. Integrity logic makes systems survivable.  
Swap in stronger field SLAM later without rewriting the whole brain — interfaces matter.

## Close the loop
See hazard → stop → reorient → replan → resume skill.  
Slope pulse turn is the keystone milestone that unlocks the rest on crater walls.

## Lunar reality
South pole craters, permanently shadowed regions, ice and ISRU — why agencies care.  
Shackleton-class walls around thirty degrees are in the mobility regime we’re building.  
Earth first. Then lunar gravity fine-tune, friction randomization, energy metrics for mission planning.  
Full granular soil sim is a later infrastructure bet — we say that out loud.

## Industry frame
Unitree made wheeled quads real and reachable.  
Pure legged and pure rover each win different games.  
Our bet: hybrid morphology, skill library, mission autonomy.

## Culture that de-risks capital
Frozen production skills.  
Published-style eval envelopes.  
One theme per training run.  
Visual acceptance tests.  
Delete failed experiments so they don’t contaminate the stack.  
Interfaces over monoliths.

## Soft ask
If you care about extreme-terrain and lunar-adjacent mobility:  
this is a product architecture — skills, nav, missions — with honest gaps and a milestone ladder.  
Partners and capital push: slope reorient, full mission autonomy on gated SLAM, then hardware.

## Milestone ladder (on screen)
1. Visually gated slope reorient  
2. Full obstacle detour loop in crater  
3. Profiles + SLAM default + dashboard  
4. Lunar-g + energy/risk metrics  
5. Go2W-class hardware path  

## Series close
Skills make motion.  
Systems make a robot.  
Missions make a product.

---

## Emergency honesty lines (if demo glitches on camera)

- “This is exactly why we freeze checkpoints and keep eval — demos are samples from a distribution.”  
- “Nav fell back to recovery; that’s the FSM doing its job.”  
- “We don’t claim steep pivot until it passes a visual gate both ways.”  

---

## Do not say

- Exact reward weights or “just increase X to 2.0”  
- “We’ve solved Shackleton” / “TRL-9” / guaranteed dates  
- “SLAM is perfect”  
- Competitor dunking  
- That spin/turn failed runs are still production  
