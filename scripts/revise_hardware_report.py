"""Build hardware report revision B from preserved revision A and run evidence.

The baseline report is intentionally preserved, not reconstructed from memory.
Exact replacements fail if the input narrative has changed and needs re-review.
"""
import argparse
from pathlib import Path
import re


def revise(original, evidence):
    text = original

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f'Expected one baseline match: {old[:100]}')
        text = text.replace(old, new)

    replace('REXMI / engineering review 01', 'REXMI / engineering review 02')
    replace('Revision A · 09 Sep 2026', 'Revision B · 09 Sep 2026 · first teleop evidence')
    replace('<a href="#loads">Loads</a>', '<a href="#loads">Loads</a><a href="#telemetry">Recorded joint loads</a>')
    replace('Why actuator selection is still open.</strong> At a hypothetical 150 mm horizontal lever arm, equal four-leg support needs 7.92 N·m per loaded joint; two-leg support needs 15.84 N·m. These are support-moment screens, not solved joint torques. They show why an “8 N·m” label alone is insufficient.',
            'The new evidence changes the actuator decision.</strong> First teleop capture: rear-right knee 15.80 Nm RMS, 22.08 Nm highest 10-second RMS, repeated 23.5 Nm clipping. A GIM8108-8 plus added reduction has a torque–speed endpoint conflict for this recorded motion. These are simulation estimates, not hardware-qualified ratings. <a href="#telemetry">Inspect the trajectories and ratio screen.</a> The 150 mm lever calculation remains an illustrative sensitivity check only.')
    replace('Labels separate requirements, source evidence and preliminary analysis. No physical torque, thermal or whole-robot acceptance testing was performed for this report.',
            'Labels separate requirements, source evidence and preliminary analysis. Revision B adds a 159.14 s teleop capture; no physical torque, thermal or whole-robot acceptance testing was performed. The 21.523 kg card is the design reference; actual randomized capture mass was 21.770 kg.')
    replace('URDF: 35.55 N·m / 20.07 rad/s; current config: 23.5 / 30',
            'URDF: 35.55 N·m / 20.07 rad/s; source config requests 23.5 / 30. Captured runtime: 23.5 N·m / 20.07 rad/s; old implicit velocity_limit request is ignored')
    replace('Resolve the exact settings for each checkpoint; caps are not continuous requirements',
            'Use resolved runtime/checkpoint settings. Wheel overspeed above its recorded ~30.1 rad/s also needs diagnosis; caps are not continuous requirements')
    replace('<section id="motors">', evidence+'\n<section id="motors">')
    replace('Does the “8 N·m” motor suffice?', 'Actuator options after the first capture')
    replace('Support moment grows with lever arm', 'Keep the lever-arm screen in its proper role')
    replace('What the screen tells us', 'What the illustrative screen tells us')
    replace('At 150 mm: 7.92 N·m on four supports; 15.84 N·m on two.',
            '150 mm was an assumed reaction-line offset, not a fixed leg length or measured policy posture. At that offset the screen is 7.92 Nm on four equal supports or 15.84 Nm on two.')
    replace('Actual posture could reduce or increase the moment arm.',
            'Actual posture changes each joint’s lever arm. Three contacts do not imply equal thirds; measured contact histories invalidate guaranteed three-leg support for this run.')
    replace('Compare synchronized demand against measured torque–speed curves.',
            'Use the new per-joint trajectories, validate the 5 ms estimate/state alignment, then compare against measured motor torque–speed curves.')
    replace('An additional external 1.5:1 stage increases module output torque by 1.5η and reduces output speed by 1.5. This is a design deviation to evaluate, not a selected solution. RMS torque is only an initial thermal proxy.',
            'The recorded RR knee endpoint screen needs N≥2.34 for a 7.5 Nm nominal module at assumed η=0.9, but N≤1.67 at 320 rpm. No selected ratio closes both. Nominal torque, peak duration, voltage and the full torque–speed curve still need qualification; finite-run RMS is only a thermal proxy.')
    replace('<h3>Checks beyond stress</h3>',
            '<h3>Transmission is a coupled design choice</h3><p>A shaft/bevel stage is a candidate, not a verified Unitree detail or automatically stiff solution. Check shaft twist θ=TL/(GJ), gear-mesh compliance, backlash, bearing thrust, housing deflection and lubrication. Added reduction reflects motor inertia approximately with N²; include it, mass, friction and elasticity in replay. Preserve link lengths first.</p><h3>Checks beyond stress</h3>')
    replace('Battery capacity follows measured duty-cycle power.',
            'Battery capacity follows measured duty-cycle electrical power. The new capture contains estimated mechanical power only; it does not close battery capacity, drive-current, regen or winding-temperature sizing.')
    replace('Capture representative policy joint/contact load histories.',
            'First load capture complete. Repeat mirrored maneuvers on measured slopes with a controlled payload and validate torque estimates/speed-limit behavior.')
    replace('Current outcome:</strong> requirements and screening are documented. Full dynamic joint sizing, battery/regen qualification, component selection, manufacturing CAD and PCB design remain future work.',
            'Current outcome:</strong> requirements, static screens and first recorded joint envelopes are documented. Knee candidate endpoint mismatch is identified; motor selection is not released. Validated actuator dynamics, full duty/terrain coverage, battery/regen qualification, manufacturing CAD and PCB design remain open.')
    replace('VISUAL REPORT: 8.7/10 · ROUND 1', 'REVISION B: 8.8/10 · ROUND 1')
    replace('Independent visual-report review: 8.7/10 in round 1 (technical 9.0, narrative/readability 8.5, coverage 8.5); source/HTML review, with browser QA by the author.',
            'Revision A independent review: 8.7/10 in round 1. Revision B independent review: 8.8/10 in round 1 (technical 9.0, narrative 8.5, traceability/coverage 9.0), including independent raw-log recomputation. This is document/analysis readiness, not hardware qualification. Revision B supersedes actuator conclusions where identified.')
    replace('<ol class="source-list">', '<ol class="source-list"><li><strong>Revision B capture:</strong> <a href="https://github.com/regmis4/rexmi_hardware/tree/main/analysis/teleop_20260909_01">numerical exports and run summary</a>; <a href="https://github.com/regmis4/rexmi_hardware/blob/main/analysis/analyze_joint_telemetry.py">reproducible analyzer</a>. Raw logs retained locally as <code>manual_20260909_01</code>; checksums are embedded in the telemetry section. Run used the rexmi_dog reskin and startup base-mass randomization, not a mounted payload. <a href="https://isaac-sim.github.io/IsaacLab/v2.2.1/_modules/isaaclab/actuators/actuator_pd.html">Implicit actuator implementation</a> explains effort-estimate semantics; installed source was also inspected.</li>')
    text = text.replace('https://isaac-sim.github.io/IsaacLab/v2.2.1/_modules/isaaclab/actuators/actuator_pd.html', 'https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.actuators.html')
    replace('REXMI · Hardware design review A', 'REXMI · Hardware design review B')
    return text


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(revise(a.baseline.read_text(), a.evidence.read_text()))
    print(a.output)
