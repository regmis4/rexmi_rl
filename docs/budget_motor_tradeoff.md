# Budget actuator tradeoff — 2026-09-10

Target: $5–7k total robot. Cost ceiling is a design constraint, not authority to discard demonstrated motion requirements. Existing HTML RS06 baseline remains reference, not purchase release.

## Finding

Not every joint needs RS06. RS02 plus an external 1.75:1 hip reduction is a credible cheaper candidate; keep RS06 direct thighs and 2:1 knees for a conservative mixed leg baseline. Four RS02 and eight RS06 cost $2,260 at advertised $145/$210 retail, before added hip transmissions. Savings versus twelve RS06: $260 gross. Whether this saves assembled cost depends on four extra transmissions, bearings and encoders.

RS01 is only $15 cheaper than RS02, but its published loaded curve is at 36 V and its operating ceiling is 48 V. RS02 supports the common 48 V architecture with a 60 V specified ceiling. Do not operate RS01 from a 48 V nominal pack whose full-charge voltage exceeds 48 V, or scale its 36 V curve to 48 V without manufacturer evidence. An extra power rail/regeneration path could erase the $60 saving across four hips.

## Actual paired-sample screening

Both captures retained after 2 s. Forward motoring only: torque × velocity >0. Ratios below are additional to integrated gearing. Assumed added-stage efficiency 95%. Published loaded points linearly interpolated, constant peak below first point, zero allowance beyond highest tabulated speed. This is a nominal-voltage screening model, not guaranteed hardware performance. Percentages are joint-time samples across four same-family joints, not percent of mission failure or stability probability.

| Family / candidate | Extra ratio | Motoring samples outside model | Interpretation |
|---|---:|---:|---|
| RS01 hip, 36 V | 1.5:1 | 0.000605% | One retained joint-time sample misses; not an exact pass |
| RS02 hip, 48 V | 1.75:1 | 0% | Credible lower-cost hip candidate |
| RS02 thigh, 48 V | 1.5:1 | 0.009524% | Fast loaded events still exceed envelope |
| RS02 knee, 48 V | 1.5:1 | 0.023501% | Better speed retention, poor sustained thermal margin |
| RS02 knee, 48 V | 2:1 | 0.083035% | Better torque/heating, loses fast knee events |

A tiny fraction outside the model is not evidence the deviation is harmless: one missed recovery impulse can cause a fall. Conversely, it is a good reason to test the cheaper hardware model in simulation rather than reject it solely from independent torque/speed maxima.

## Concrete cheaper experimental configuration

- Four RS02 hips: 1.75:1 external reduction (e.g. 24:42 tooth count concept), total 13.5625:1.
- Four RS02 thighs: 1.5:1 external reduction (24:36 concept), total 11.625:1.
- Four RS02 knees: 2:1 external belt reduction (24:48 concept), total 15.5:1.
- Twelve modules: **$1,740** at $145 advertised retail. This is a replay candidate, not the selected performance-preserving BOM.
- Knees: 23.5 Nm joint cap maps to 12.37 Nm module torque; worst-run RMS 8.31 Nm and worst 10-second RMS 11.62 Nm. These exceed RS02's 7 Nm nominal reference under specified cooling. Do not claim continuous operation from its 17 Nm peak rating.
- At 2:1, max observed knee speed demands 383.72 module rpm; RS02 loaded curve supports only roughly 4.1 Nm there, not the conservative 12.37 Nm rectangle. Actual paired trajectory misses are recorded in the screen.
- Prefer one symmetric motor/ratio per joint family. Do not assign a smaller left/right motor because the one-sided crater loaded the opposite side more; mirrored traversals can swap governing joints.
- RS02 module mass is 380 g; versus RS06 621 g, twelve modules save 2.892 kg before extra hip/thigh transmissions. That can lower loads, but requires updating CAD mass/COM/inertia and replaying; linear mass scaling is not validation.

## Whole-robot allocation — target, not vendor quotation

| Bucket | Planning allocation USD |
|---|---:|
| Actuators and wheel drives | 2,500 |
| External transmissions and bearings | 550 |
| Frame, machined parts and covers | 700 |
| Battery, protection, power distribution | 500 |
| Onboard compute | 650 |
| Sensors | 350 |
| Harness, CAN, connectors and emergency stop | 300 |
| Fasteners and miscellaneous hardware | 150 |
| Contingency including shipping/tax/rework | 800 |
| **Total target** | **6,500** |

These are spend limits, not researched component prices or a closed BOM. A capable wheel-drive selection is still missing. Owned compute/sensors/ODrives may free budget, but have not been assumed free. Labor/tools excluded. A $5k build is more dependent on owned equipment and inexpensive fabrication; $7k leaves more room. Keep all hardware purchases pending until actual quotes and wheel design close the allocation.

## Sourcing and decision

- First request landed quantity-12 pricing for RS02 and RS06 from manufacturer/authorized channels; no request sent here. Manufacturer July 2026 domestic list is ¥699 / ¥849 respectively, distinct from $145 / $210 international retail. No exchange-rate conversion or landed cost is implied.
- AliExpress is a sourcing channel, not a different torque-speed capability. Search surfaced RS02 listings but did not establish a cheaper, current landed quote with the exact variant and warranty. No unverified listing is promoted into the BOM.
- GIM8108-8 remains a comparison, not an established knee replacement: the previously checked 7.5 Nm /320 rpm variant cannot meet the conservative knee RMS and speed endpoint screens simultaneously with fixed extra gearing.
- Next useful implementation: separate budget replay configuration with RS02 curves, gear mapping, transmission inertia/compliance and thermal model; preserve existing teleop baseline. Re-run the same course and mirrored motions. Only user-accepted new performance can relax the original trajectory requirement.
- Do not purchase a full set based on this screening. A one-leg prototype and thermal test can decide whether the cheaper architecture is actually economical.

## Traceability

- scripts/screen_budget_motors.py → motor_selection_evidence/budget_motor_screen.csv, 42 candidate/family screens.
- Original capture hashes in motor_selection_evidence/manifest.json. Raw captures unchanged.
- Official prices: https://robstride.com/ (checked 2026-09-10).
- Official RS01 voltage/details: https://robstride.com/products/robStride01 (older page says 6 Nm nominal; latest July 2026 PDF says 7 Nm under specified heatsink conditions; use exact revision in procurement).
- Official loaded curves: RobStride/Product_Information, July 13 2026 specification PDF, RS01 page 8 and RS02 page 12. https://github.com/RobStride/Product_Information
- GIM variant reference: https://steadywin.cn/en/pd.jsp?id=133
