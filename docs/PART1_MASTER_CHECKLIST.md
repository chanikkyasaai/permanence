# MASTER CHECKLIST: WHAT NEEDS TO HAPPEN FOR PART 1

## Files Already Prepared (✓ Done)

| File | Purpose | Status |
|------|---------|--------|
| `PART1_QUICK_SUMMARY.md` | 1-page reference guide for venue | ✓ READY |
| `PART1_DEVELOPMENT_TRAINING_CHECKLIST.md` | Detailed step-by-step instructions | ✓ READY |
| `generate_curves.py` | Curve generation after training | ✓ READY |
| `BLOG_POST_TEMPLATE.md` | Storytelling framework | ✓ READY |
| `training/train.py` | Training script | ✓ READY |
| `training/config.yaml` | Optimized config (1500 episodes) | ✓ READY |
| `training/warmup_traces.jsonl` | SFT warmup data (20 examples) | ✓ READY |
| `permanence/env.py` | Core environment | ✓ READY |

---

## PART 1: DEVELOPMENT & TRAINING BREAKDOWN

### What Happens in PART 1
**At venue: 11:30 AM - 8:00 PM (8.5 hours)**

PART 1 is about **generating evidence that your environment actually teaches agents something.**

---

## WHAT YOU NEED TO DO (Concrete Tasks)

### PRE-VENUE (Before you leave today)

**Task 1.1: Verify repo is in good state**
```bash
cd c:\Users\Hp\OneDrive\Desktop\meta
git status  # Should show nothing uncommitted
git log -1  # Last commit: "Add OpenEnv deployment files..."
```
Expected: No uncommitted changes, repo clean

**Task 1.2: Verify dependencies are specified**
```bash
cat pyproject.toml | grep -A 10 dependencies
```
Expected: Lists torch, transformers, trl, unsloth, datasets, peft

**Task 1.3: Verify training config is correct**
```bash
cat training/config.yaml
```
Expected: `total_episodes: 1500`, `group_size: 8`, `load_in_4bit: true`

---

### AT VENUE: PHASE 1 (11:30 AM - 12:00 PM) — GPU Setup

**Task 2.1: Get GPU access**
- Find venue staff
- Get SSH credentials or Colab link
- **CRITICAL:** Confirm GPU type (A100, RTX 4090, H100, etc.)
- If NO GPU: Escalate immediately to L2 mentor

**Task 2.2: Verify CUDA works**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB')"
```
Expected: Should print GPU name and memory (e.g., "A100" and "40GB")

**Task 2.3: Clone repo and install dependencies**
```bash
git clone https://github.com/chanikkyasaai/permanence
cd permanence
pip install -e .
pip install torch transformers trl unsloth datasets peft
```
Expected: No errors, all packages install successfully

**Task 2.4: Verify environment works**
```bash
python -c "from permanence.env import PermanenceEnv; print('✓ OK')"
```
Expected: Prints "✓ OK"

**By 12:00 PM: You should have GPU ready, repo cloned, dependencies installed, environment verified.**

---

### AT VENUE: PHASE 2 (12:00 PM - 7:30 PM) — Training Execution

**Task 3.1: START TRAINING (single command)**
```bash
python -m training.train --config training/config.yaml
```

**That's it. Press Enter. Training runs for 7 hours unattended.**

**What happens next:**
- Minutes 0-1: Model loading
- Minutes 1-3: Data loading
- Minutes 3-420: Training (1,500 episodes × ~0.17 min/episode)
- Every 100 episodes: Progress printed to console
- Output: `permanence_output/training_log.json` with all metrics

**You can relax, walk around, eat, prepare for Part 2. Just don't close the terminal.**

**Checkpoint:** Every 500 episodes, a checkpoint is saved. If it crashes at episode 1400, you can resume.

---

### AT VENUE: PHASE 3 (7:30 PM - 8:00 PM) — Post-Training Verification

**Task 4.1: Generate training curves**
```bash
python generate_curves.py
```
Expected: Creates `results/training_curves.png` (4-panel plot)

**Task 4.2: Verify curves look good**
- Open `results/training_curves.png`
- Check Panel 1 (Reward): Should trend **upward** (from negative to positive)
- Check Panel 2 (Loss): Should trend **downward** (convergence)
- Check Panel 3 (Catastrophe): Should trend **downward** (improvement)
- Check Panel 4 (Accuracy): Should trend **upward** (improvement)

If curves look wrong: Check training_log.json for errors

**Task 4.3: Verify model loads**
```bash
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('./permanence_output/final_model'); print('✓ Model loads')"
```
Expected: Prints "✓ Model loads"

**Task 4.4: Commit results**
```bash
git add permanence_output/training_log.json results/training_curves.png results/training_summary.txt
git commit -m "Training complete: 1500 episodes, reward improvement verified"
```
Expected: Commit succeeds, files tracked

**By 8:00 PM: You have training curves, metrics, and proof that the environment works.**

---

## DELIVERABLES AT END OF PART 1

By 8:00 PM, you will have:

```
permanence_output/
├── training_log.json              ← 1,500 episodes of metrics
├── final_model/                   ← Trained weights
│   └── pytorch_model.bin
└── checkpoint_*

results/
├── training_curves.png            ← ⭐ JUDGES WANT THIS
├── training_summary.txt           ← Numerical metrics
└── training_comparison.md

Git commits with all artifacts tracked
```

---

## SUCCESS CRITERIA FOR PART 1

✅ You've completed PART 1 if:

- [ ] Training ran for 7 hours without crashing
- [ ] permanence_output/training_log.json exists with 1,500 episodes
- [ ] results/training_curves.png exists and shows improvement
- [ ] Reward curve trending upward
- [ ] Catastrophe rate trending downward (from ~43% to <20%)
- [ ] Prediction accuracy trending upward (from ~31% to >50%)
- [ ] Trained model loads successfully
- [ ] All results committed to git

---

## WHAT COMES AFTER PART 1 (PART 2)

Once PART 1 is complete (8:00 PM), you'll have 9 hours until deadline (5:00 PM next day) to do PART 2:

**PART 2 Tasks:**
1. Write mini-blog or record <2min video explaining results
2. Update README with storytelling arc + curve + links
3. Push to HuggingFace Space
4. Update GitHub with final links
5. Submit Google Form

(PART 2 checklist will be provided separately once PART 1 is done)

---

## KEY FACTS

**PART 1 is the bottleneck.** Everything depends on getting GPU training to work.

**Judges explicitly state:** "At minimum, loss and reward plots from a real run."

**Right now:** You have 0/20 on "Training Evidence" criterion. After PART 1: You'll have 7/20.

**The difference:** Disqualification vs. Contention.

**What must happen:** Train for 7 hours, generate curves, commit results.

**Contingency:** If GPU fails, you can still explain the technical architecture to judges. But curves are what wins.

---

## IMMEDIATE NEXT STEPS

### Today (Before Venue):
- [ ] Print or bookmark `PART1_QUICK_SUMMARY.md` (2 pages, reference at venue)
- [ ] Review `PART1_DEVELOPMENT_TRAINING_CHECKLIST.md` (detailed steps)
- [ ] Verify training/config.yaml one more time
- [ ] Make sure laptop has repo cloned locally (backup copy)

### At Venue (11:30 AM):
- [ ] Find GPU
- [ ] Follow PART1_QUICK_SUMMARY.md steps 1-3
- [ ] Start training at 12:00 PM
- [ ] Follow post-training steps at 7:30 PM
- [ ] Curves ready by 8:00 PM

**That's the entire PART 1 plan. Nothing more complicated than that.**
