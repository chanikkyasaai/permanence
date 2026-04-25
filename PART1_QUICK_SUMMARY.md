# PART 1: WHAT YOU NEED TO DO
## One-Page Summary

---

## **BEFORE YOU LEAVE FOR VENUE (Next 2 hours)**

### Verify these files exist and are committed:
```
✓ training/train.py — the training script
✓ training/config.yaml — training configuration  
✓ training/warmup_traces.jsonl — SFT data (20 examples)
✓ training/evaluate.py — holdout evaluation
✓ permanence/env.py — the core environment
✓ generate_curves.py — curve generation script (in root dir)
✓ BLOG_POST_TEMPLATE.md — your blog post skeleton
```

**Status Check:**
```bash
git status  # Should show nothing uncommitted
git log -1  # Last commit should be "Add OpenEnv deployment files..."
```

---

## **AT VENUE: 11:30 AM - 7:30 PM**

### STEP 1: Get GPU & Set Up (30 minutes, 11:30 AM - 12:00 PM)

1. Find venue staff, get GPU access
2. SSH to GPU machine
3. Verify GPU:
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name(0))"
   ```
4. Clone repo:
   ```bash
   git clone https://github.com/chanikkyasaai/permanence
   cd permanence
   ```
5. Install dependencies:
   ```bash
   pip install -e .
   pip install torch transformers trl unsloth datasets peft
   ```
6. Quick sanity check:
   ```bash
   python -c "from permanence.env import PermanenceEnv; print('✓ OK')"
   ```

**SUCCESS:** Should print "✓ OK" with no errors

---

### STEP 2: START TRAINING (1 command, 12:00 PM)

```bash
python -m training.train --config training/config.yaml
```

**Then WAIT 7 hours.** The training will:
- Run 1,500 episodes
- Generate permanence_output/training_log.json
- Save trained model to permanence_output/final_model/
- Print progress every 100 episodes

**You don't need to babysit it,** but check every hour that it's still running (monitor GPU usage).

---

### STEP 3: Post-Training Verification (30 minutes, 7:30 PM - 8:00 PM)

Once training finishes:

1. **Generate curves:**
   ```bash
   python generate_curves.py
   ```
   Creates: `results/training_curves.png`

2. **Check the curves exist:**
   ```bash
   ls -la results/training_curves.png
   ```

3. **Check model loads:**
   ```bash
   python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('./permanence_output/final_model'); print('✓ Model OK')"
   ```

4. **Commit results:**
   ```bash
   git add permanence_output/training_log.json results/
   git commit -m "Training complete: 1500 episodes"
   ```

**SUCCESS CRITERIA:**
- ✅ results/training_curves.png exists
- ✅ Curves show reward going UP (positive trend)
- ✅ Catastrophe rate going DOWN
- ✅ Model loads without error

---

## **OUTPUTS YOU'LL HAVE**

At 8:00 PM, you'll have:

| File | Purpose |
|------|---------|
| `permanence_output/final_model/` | Trained model weights |
| `permanence_output/training_log.json` | All 1500 episode metrics |
| `results/training_curves.png` | **Publication-quality 4-panel plot** ← JUDGES WANT THIS |
| `results/training_summary.txt` | Numerical metrics |
| Git commit | Everything tracked |

---

## **WHY THIS MATTERS**

The judging criteria explicitly state:
> "Showing Improvement in Rewards (20%): **Is there observable evidence of training progress? Reward curves, metrics, or before/after behavior** — anything that proves the agent learned something."

**You now have this evidence.** That's 20% of the grade locked in.

Without these curves: 0/20  
With curves showing improvement: 7/20

**This is the difference between disqualification and contention.**

---

## **IF SOMETHING BREAKS**

| Problem | Fix |
|---------|-----|
| GPU not available | Ask venue staff for alternative GPU |
| Out of memory | Edit config.yaml: change `group_size: 8` to `group_size: 4`, restart training |
| Training stuck/very slow | Check if GPU is shared; ask mentor |
| Model won't load | Verify permanence_output/final_model/ has files; may be corruption |

**In all cases:** Escalate to L2 mentor immediately. Don't wait.

---

## **TIMELINE SUMMARY**

| Time | Task | Duration |
|------|------|----------|
| 11:30 AM - 12:00 PM | GPU setup + dependency install | 30 min |
| 12:00 PM - 7:30 PM | **TRAINING RUNS** (you can relax/prepare for Part 2) | 7 hours |
| 7:30 PM - 8:00 PM | Verify output + generate curves | 30 min |
| **8:00 PM** | **PART 1 COMPLETE** ✓ | |

**You're done by 8:00 PM. Part 2 (demo & submission) starts then.**

---

## **THAT'S IT FOR PART 1.**

Everything is set up. The training script works. The environment is tested. The curve generator is ready.

Just get GPU access, copy-paste the training command, and wait.

The curves will show judges that your environment actually teaches agents something measurable.

---

**Questions before you leave? Ask now. Everything else is at the venue.**
