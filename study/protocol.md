# MogScan user study: protocol

**Goal.** Collect real evidence for the Startup-track rubric item "evidence of real-world problem-solving".
Measure four things: usability (SUS), perceived accuracy and usefulness of the explanations, wellbeing impact,
and willingness to pay. The session logs also give an objective measure of score stability and frame rate.

**Who:** 8–15 adults (18+), recruited from friends, classmates, and dorm/club members.
Run one participant at a time.

**Time per session:** about 10 minutes per participant.

**Equipment:**
- A laptop running MogScan with `?study` added to the URL (shows the Record button), via `npm run dev` or the deployed URL
- A USB webcam at eye level, 1–1.5 m away
- Even, front-facing light
- The Google Form open on a phone

## Steps
1. **Consent (2 min).** Read `consent_script.md` aloud. Anyone can stop at any time.
   Do not scan anyone who hasn't agreed, including people walking behind the camera.
2. **Solo scan (3 min each).** The participant sits alone in frame. Press **● Record**.
   Ask them to look at the camera and hold still for about 10 seconds. Then ask them to slowly turn their head
   left and right; this tests pose gating. Press **■ Stop & save log**. Let them read their card and expand
   "Show all 143 inputs and their weights".
3. **Photo upload (2 min).** The participant uploads a front-facing photo of themselves from their phone or laptop
   with **Upload photo**, compares the result with their live score, then returns with **Live camera**.
4. **Survey (5 min).** Each participant fills out the Google Form on their own phone (questions in `survey.md`).
5. **Debrief (1 min).** Remind them that the score reflects the biases of one dataset (60 young raters scoring
   Asian and Caucasian faces only), not their worth.

## Data handling
- No images or video are stored. The app is on-device, and the session logs contain only numbers.
- Save logs to `study/logs/` and export the Google Form responses to `study/responses.csv`.
- Do not write names in either. Use participant codes P01, P02, … in the form's first question.

## Analysis
```
../training/.venv/bin/python analyze_survey.py      # reads responses.csv
../training/.venv/bin/python analyze_logs.py        # reads logs/*.json
```
Both write tables into `study/out/` and figures into `writeup/figures/`. The write-up and slides use those numbers.
