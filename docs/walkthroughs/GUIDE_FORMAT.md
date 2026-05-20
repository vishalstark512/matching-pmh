# Walkthrough guide format

Every numbered walkthrough is a **full adoption guide**, not a paper task note. Use this checklist when reading or contributing.

---

## Standard sections (in order)

1. **At a glance** — table: estimator, stack, script, time to run  
2. **Who this is for** — when to use / when to pick another walkthrough  
3. **Prerequisites** — install extras, hardware, data layout  
4. **Your deployment shift sentence** — 2–3 concrete examples (what changes at deploy, not the label)  
5. **What the example script does** — line-by-line map to library calls  
6. **Step-by-step on your code** — copy-paste with `YOUR_*` placeholders  
7. **Run the example** — command, env vars, expected output  
8. **Adaptation worksheet** — table: Example → Your project  
9. **Verify success** — checklist + metrics  
10. **Controls** — link to WT 8, arm table  
11. **Common mistakes** — bullet list  
12. **Next steps** — related walkthroughs  

---

## Runnable scripts

Each walkthrough links to one primary script under `examples/`. Scripts must run without paper datasets (synthetic or tiny bundled JSONL only).

---

## Adoption banner (required on each WT)

Each walkthrough opens with an **Adopt PMH first** tip: link [ADOPT.md](../../ADOPT.md), golden path, `pmh-train route --task`, Step 5.

## Daily AI map

[DAILY_AI_USE.md](DAILY_AI_USE.md) — which walkthrough matches which job.
