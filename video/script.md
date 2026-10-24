# MogScan demo video: script (target 6–7 min)

Record the screen with QuickTime (File → New Screen Recording) or OBS at 1080p.
Record yourselves with the USB webcam pointed at 2–3 people. Then run `./assemble.sh` to stitch the clips.

| # | Time | Clip file | On screen | Voice-over (Ben = B, Cyril = C) |
|---|------|-----------|-----------|----------------------------------|
| 1 | 0:00–0:20 | `01_hook.mov` | Cold open: one person sits down, their card pops up, and they open the pillar sub-scores. | *(no VO, just their reaction)* |
| 2 | 0:20–0:50 | title card (auto) | "MogScan: real-time, explainable PSL" | B: "Face-rating apps have tens of millions of downloads, and Omoggle just took over Twitch. But they turn your face into a competition or a vague category score, they won't tell you *why*, and most of them upload your selfie to a server." |
| 3 | 0:50–1:30 | `02_problem.mov` | Screenshots of Omoggle, UMAX, LooksMax AI paywalls | C: "We looked at how the current apps work. Omoggle pits you against strangers 1v1 and is easily thrown off by camera angles. UMAX and LooksMax lock your results behind a $4–10 a week paywall. None of them show that their score matches what real people think." |
| 4 | 1:30–3:30 | `03_demo_solo.mov` | Ben alone. Hold still while the scan bar fills and the score locks. Open the card: point at the Harmony / Dimorphism / Angularity / Softmax pillars, then "Show all 143 inputs and their weights". Turn your head: the score stays locked. Click Rescan to scan again. | B: walk through the PSL, the tier, the top-x%, and two strengths and weaknesses with real numbers. Explain that it scans for ~2 seconds and then locks, so the score does not jitter. |
| 5 | 3:30–4:30 | `04_demo_upload.mov` | Click Upload photo and pick a front-facing photo; the card appears. Click Live camera to go back. Camera picker dropdown → USB camera. | C: "Same model on a photo or a live USB camera. It runs at around 30 fps on a laptop, there are no battles or leaderboards, just your own breakdown, and no pixels leave the machine." |
| 6 | 4:30–5:30 | `05_how.mov` | Slides 6–7 (architecture + results table) | B: MediaPipe 478 landmarks → frontalization → 143 features (geometry, 3D depth, skin, expression) → ridge model; MobileNetV3 CNN; fusion. Read the actual test-set Pearson r for each model from `training/out/results.json`. |
| 7 | 5:30–6:15 | `06_traction.mov` | Slide with the user-study numbers + quotes | C: report the real SUS / NPS / % agree from `study/out/survey_summary.json`. |
| 8 | 6:15–6:45 | `07_close.mov` | Slide: roadmap + QR code to the live URL | B: "Next up: progress tracking, a side-profile scan, and a mobile app. Try it at the QR code." |

**Tips:**
- Light from the front. A window behind people breaks detection.
- Mute any music in the screen recording. Assemble the music separately.
- Only film people who signed the consent form.
