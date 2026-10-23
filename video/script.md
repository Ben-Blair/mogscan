# MogScan demo video: script (target 6–7 min)

Record the screen with QuickTime (File → New Screen Recording) or OBS at 1080p.
Record yourselves with the USB webcam pointed at 2–3 people. Then run `./assemble.sh` to stitch the clips.

| # | Time | Clip file | On screen | Voice-over (Ben = B, Cyril = C) |
|---|------|-----------|-----------|----------------------------------|
| 1 | 0:00–0:20 | `01_hook.mov` | Cold open: three friends step into frame and their cards pop up. The MOG BATTLE verdict appears. | *(no VO, just their reactions)* |
| 2 | 0:20–0:50 | title card (auto) | "MogScan: real-time, multi-person, explainable PSL" | B: "Face-rating apps have tens of millions of downloads, and Omoggle just took over Twitch. But they rate one face at a time, they won't tell you *why*, and most of them upload your selfie to a server." |
| 3 | 0:50–1:30 | `02_problem.mov` | Screenshots of Omoggle, UMAX, LooksMax AI paywalls | C: "We looked at how the current apps work. Omoggle is 1v1 and is easily thrown off by camera angles. UMAX and LooksMax lock your results behind a $4–10 a week paywall. None of them show that their score matches what real people think." |
| 4 | 1:30–3:30 | `03_demo_solo.mov` | Ben alone. Show Guides on: the canthal-tilt lines, thirds, jaw. Open the card, then "Show all 15 measurements". Turn your head: "face the camera" appears and the score holds steady. | B: walk through the PSL, the tier, the top-x%, and two strengths and weaknesses with real numbers. Explain that the score freezes when you turn your head (pose gating). |
| 5 | 3:30–4:30 | `04_demo_group.mov` | 2–3 people. Turn on Mog battle, switch to Brutal mode, take a Snapshot. Camera picker dropdown → USB camera. | C: "It handles up to six people at once. Each person gets their own ID and smoothed score, all at around 30 fps on a laptop, and no pixels leave the machine." |
| 6 | 4:30–5:30 | `05_how.mov` | Slides 6–7 (architecture + results table) | B: MediaPipe 478 landmarks → frontalization → 15 metrics → ridge model; MobileNetV3 CNN; fusion. Read the actual test-set Pearson r for each model from `training/out/results.json`. |
| 7 | 5:30–6:15 | `06_traction.mov` | Slide with the user-study numbers + quotes | C: report the real SUS / NPS / % agree from `study/out/survey_summary.json`. |
| 8 | 6:15–6:45 | `07_close.mov` | Slide: roadmap + QR code to the live URL | B: "Next up: progress tracking, a side-profile scan, and a streamer overlay for OBS. Try it at the QR code." |

**Tips:**
- Light from the front. A window behind people breaks detection.
- Mute any music in the screen recording. Assemble the music separately.
- Only film people who signed the consent form.
