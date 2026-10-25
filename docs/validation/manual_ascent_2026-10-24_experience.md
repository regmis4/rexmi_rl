## User experience and route-selection logic

The following is the user's account of the successful 24 October manual ascent,
lightly edited for readability. It records intent; measured outcomes are in the
report above. It supersedes interpreting the earlier green sketch as the desired
route itself.

### Path design — user's account

> I knew climbing was going to be hard. I looked for the darkest sections of the
> costmap that offered the most elevation gain. I made sure that reaching those
> sections would leave a path to the exterior, rather than a dead end surrounded
> by massive costs and terrain to avoid.
>
> After gaining about 2 m of elevation, I needed to turn and use a small ring of
> lower-cost terrain to find an opening to the exterior. I followed that ring,
> then tried to get out. That left only a small section with a large elevation
> gain. Angling the robot correctly allowed a gentler diagonal approach through
> that section.

### Driving experience — user's account

> I let the robot do its thing. I commanded forward to start the climb, then
> stopped sending commands once it was climbing. I would intervene after perhaps
> 5–10 seconds of failed climbing, turning slightly toward darker patches to try
> a gentler diagonal slope. If the robot climbed a little and then slipped back,
> I tried turning: it already knew how to move forward.

“No command” here means zero commanded forward and yaw velocity while the learned
policy continues running. The 5–10 seconds describes **failed net progress**, not
a deadline for stopping a productive climb. The approximately 2 m gain describes
this route; it is not a universal altitude checkpoint.

### What the recording supports

- First exterior-radius crossing occurred 128.98 s after the first command,
  with no reset or policy change. Final position was `(5.168, 11.894)` m.
- Net body-height gain to the peak was 3.874 m from the settled start. Body
  height is distinct from ground elevation and body tilt from terrain slope.
- Zero-input intervals at 33.76–52.52 s, 54.72–68.86 s and 113.34–121.46 s gained
  0.763 m, 0.550 m and 0.348 m of body height, respectively.
- From first command to the exterior crossing, the input was forward for 27.66 s,
  turn-only for 26.86 s and zero for 74.46 s. No reverse command was used in this
  capture; it does not establish that reversing is unnecessary elsewhere.
- The measured route climbs the western corridor, traverses the elevated ring,
  then climbs toward the northeastern exterior. It is a useful executable
  baseline against which to judge route geometry and command persistence.
- Different commands occurred at different terrain and posture states. This
  demonstrates productive zero-input climbing, not that zero input is always
  better than forward, or that a wheel-speed spike proves loss of traction.

### Planning and guardrail design brief

Status: requirements derived from this demonstration. This reporting update does
not modify or validate the autonomous planner or controller.

| Decision | Required behavior | Check against this baseline |
| --- | --- | --- |
| Select an ascent corridor | Prefer useful terrain-height gain through low-cost, gentler sections. Evaluate local wheelbase-scale climbing demand and sustained exposure; a short difficult climb must not be hidden by long easy travel. | Show height gain, local grade and cost by section alongside the complete route. Avoid rewarding raw upward slope or adding a negative elevation bonus that creates search cycles. |
| Check onward access | Before committing to an elevated checkpoint, retain a feasible continuation to an exterior opening, including the eventual connecting climb. Test the complete connection, not merely proximity of two search fronts. | Demonstrate where the onward path crosses the rim; reject attractive pockets with no supported continuation. Connectivity remains a geometric prediction, not a guarantee of traction. |
| Use the elevated ring | Allow contour travel with little immediate height gain when it leads to a better exit climb. Select natural changes in terrain or approach as checkpoints. | Do not require monotonically increasing height, a heading directly toward the final goal, or a hard-coded turn after 2 m of gain. |
| Shape the final climb | Evaluate gentle diagonals using directional grade, cross slope, clearance and onward connection. Compare the hard climb sections explicitly. | A diagonal is useful only when it improves the complete approach; preserve footprint and established capability checks. |
| Execute productive climbing | Use the teleop command interface: position, initiate forward movement, release on observed response, and continue zero input while net uphill/route progress persists. | Do not brake, re-launch or replan simply because zero input has lasted several seconds. Measure ground/route advancement as well as noisy body Z. |
| Detect a failed attempt | Observe net progress and repeated gain-then-backslide over a bounded window. Use the user's 5–10 s as an initial diagnostic range to validate, not a new fixed timer. | Productive slow climbs renew patience; changing commands does not. Preserve the existing mission deadline and recovery accounting until changes are explicitly tested. |
| Adapt heading | When stalled or slipping, allow a committed small left/right correction toward a supported gentler approach, including on a slope when physically justified. Reassess the response before another correction. | Separate routine climb turns from necessary recovery turns. Avoid a blanket slope-turn veto, tick-to-tick oscillation, or exact-path tracking that forbids useful diagonal motion. |
| Keep physical protections meaningful | Preserve STOP/manual takeover, collision, drop and tip protections, existing capability limits and slope speed ceilings. Treat costs as ranking information. | A costly cell or planning rejection is not a physical failure. Do not infer universal turn permission from this one success or treat a contour as a flat turning platform. |
| Compare fairly | Generate routes independently from the same map and matched endpoints. Use the recorded route and command response only for evaluation and behavioral evidence. | No supplied human waypoints or timed command replay in an autonomous success claim. Compare completion, hard-section demand, progress, recovery and travel cost together. |

The practical behavioral sequence is **position → initiate climb → release while
progress continues → small approach correction after measured failure → retry or
reposition**. Direction and corrections depend on current observations; the
recorded button timings are evidence, not an execution script.

Next implementation should identify which current objective terms or vetoes
contradict these requirements, change the smallest responsible component, and
review a bounded test before another physical trial. A single successful manual
run is a baseline, not a calibrated probability model or autonomous qualification.
