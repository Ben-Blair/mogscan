# Google Form questions

Create a Google Form with these questions **in this order**. Start each question's title with the bracketed
code exactly as written. `analyze_survey.py` finds columns by those codes, so the rest of the wording
can change.

## Section 1: About you
- **[PID]** Participant code (given by the researcher, e.g. P03). *Short answer*
- **[CONSENT]** I am 18+ and agreed to take part. *Checkbox*
- **[AGE]** Age range. *Multiple choice: 18–20, 21–23, 24–26, 27+*
- **[PRIOR]** Have you used a face-rating / looksmaxxing app before (e.g. UMAX, LooksMax AI)? *Multiple choice: Yes, No*
- **[PRIORWHICH]** If yes, which ones? *Short answer, optional*

## Section 2: System Usability Scale
*Linear scale 1 (Strongly disagree) to 5 (Strongly agree)*
- **[SUS1]** I think that I would like to use this system frequently.
- **[SUS2]** I found the system unnecessarily complex.
- **[SUS3]** I thought the system was easy to use.
- **[SUS4]** I think that I would need the support of a technical person to be able to use this system.
- **[SUS5]** I found the various functions in this system were well integrated.
- **[SUS6]** I thought there was too much inconsistency in this system.
- **[SUS7]** I would imagine that most people would learn to use this system very quickly.
- **[SUS8]** I found the system very cumbersome to use.
- **[SUS9]** I felt very confident using the system.
- **[SUS10]** I needed to learn a lot of things before I could get going with this system.

## Section 3: MogScan specifically
*Linear scale 1 (Strongly disagree) to 5 (Strongly agree)*
- **[ACC]** The overall score felt accurate.
- **[STABLE]** The score stayed consistent while I held still.
- **[EXPLAIN]** I understood why I got the score I got.
- **[FEATURES]** The list of strong/weak features (canthal tilt, philtrum, etc.) was useful.
- **[PILLARS]** The sub-scores (Harmony, Dimorphism, Angularity, Softmax) were useful.
- **[PRIVACY]** Knowing it runs on-device (nothing uploaded) made me more comfortable.
- **[BETTER]** *(only if PRIOR = Yes)* MogScan was better than other face-rating apps I've used.
- **[FEELBAD]** Using MogScan made me feel worse about my appearance.
- **[NPS]** How likely are you to recommend MogScan to a friend? *Linear scale 0–10*
- **[PAY]** What would you pay for a version with progress tracking and a glow-up plan? *Multiple choice:
  Nothing, $1–3 one-time, $3–5/month, $5–10/month, More than $10/month*

## Section 4: Open-ended
- **[LIKE]** What did you like most? *Paragraph*
- **[CHANGE]** What would you change or add? *Paragraph*
- **[USE]** When or where would you actually use this? *Paragraph*

After the sessions: Responses → Download responses (.csv) → save as `study/responses.csv`.
