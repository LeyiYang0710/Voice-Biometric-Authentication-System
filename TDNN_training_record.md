# TDNN experiment record

Updated: 27 October 2026, Australia/Sydney.

## 1 Recording preparation

| Member | Number of sessions |
|---|---:|
| member02 | 3 |
| member03 | 2 |
| member04 | 1 |

## 2 Improvement process

All formal TDNN runs used the same Mini LibriSpeech split: 28 speakers, 1210 training recordings, 136 validation recordings, and 136 test recordings. Recordings shorter than three seconds were excluded. Training used random three-second crops; validation used centered three-second crops. Team recordings did not update network weights. Public test and reserved dev-clean-2 data were not used.

The fixed training settings were seed 3601, Adam, learning rate 0.001, and batch size 16. The small TDNN uses 40-dimensional log-Mel features with temporal mean subtraction, convolution channels 64/64/128, mean and standard-deviation pooling, and a 64-dimensional embedding before the 28-class classification head. The original configuration has 68,700 parameters. No VAD, noise augmentation, or project-designed waveform denoising filter was added.

| Step | Change | What stayed fixed |
|---|---|---|
| Smoke check | Three optimizer updates to check that training and checkpoint reload worked | Not a formal performance result |
| Original baseline | Train the linear classifier for 10 epochs | Original architecture and features |
| Embedding diagnosis | Confirm the intended 64-dimensional layer, evaluation mode, normalization, and fresh inference consistency | Network weights |
| Mean subtraction | Subtract the public validation embedding mean before normalization | Network weights and recording protocol |
| Cosine classifier | Normalize embeddings and classifier weights; use logits scaled by 16; train for 100 epochs | Architecture, data, and feature extraction |
| Multiple windows | Replace one centered crop with normalized averaging across three-second windows | Existing checkpoints; rebuild both enrolment and query vectors |
| AM-Softmax margin 0.1 | Subtract 0.1 from the target cosine during training; train for 100 epochs | Cosine setup and other training settings |
| AM-Softmax margin 0.2 | Increase only the training margin; train for 100 epochs | Other settings and member evaluation protocol |
| Longer isolated trials | Train margins 0.1 and 0.2 independently for 200 and 500 epochs; complete margin 0.2 at 1000 epochs | Seed, optimizer, architecture, data, and aggregation |
| Margin 0.3 trials | Train independent 200- and 500-epoch runs | All settings except margin and stated epoch budget |

AM-Softmax uses `16 * (cosine - margin)` for the true training class and `16 * cosine` for other classes. Validation and inference remain margin-free. This is an additive cosine margin, not an angular margin. Classifier bias is unused and frozen in cosine mode. The linear classifier remains available for older checkpoints; regression checks confirmed unchanged old-model outputs.

Each isolated training run starts from random initialization. Longer runs do not resume shorter-run checkpoints. The selected checkpoint minimizes public validation loss; it need not come from the final epoch or the epoch with highest validation accuracy. Earlier models, templates, and reports were preserved.

The margin 0.1 1000-epoch run was skipped under the agreed rule: when correct counts tie, prefer the 200-epoch checkpoint if its minimum score gap exceeds the 500-epoch checkpoint. For margin 0.2 the 500-epoch minimum gap improved, so its 1000-epoch trial proceeded.

CAM++ is the unchanged upstream pretrained control. The small-model implementation, training changes, and diagnostic scripts are AI-assisted project code; trained TDNN weights started from random initialization.

## 3 Successful and unsuccessful attempts

### 3.1 Embedding overlap and layer checks

Initial similarity to the member02 template was 0.930665 and 0.910242 for two member02 queries, versus 0.916957 and 0.864746 for two member03 queries. An impostor exceeded a genuine score. Inspection found no wrong-layer, wrong-vector-file, evaluation-mode, or normalization error. The working explanation was insufficiently discriminative embeddings for unseen members, rather than a confirmed extraction bug.

### 3.2 Mean subtraction was not adopted

The mean of the raw embeddings from 136 public validation recordings was subtracted before normalization. The same transformation was applied to enrolment and query embeddings, without changing network weights. This was compared on ten member02 session02 queries and ten member03 session01 impostor queries against the member02 template.

| Statistic | Original | Validation-mean subtraction |
|---|---:|---:|
| Mean genuine score | 0.936870 | 0.834327 |
| Mean impostor score | 0.859338 | 0.666229 |
| Mean score gap | 0.077531 | 0.168097 |
| Genuine score higher in all 100 genuine/impostor pair comparisons | 84/100 | 79/100 |

The average gap increased, but pair ordering worsened. Mean subtraction was not adopted. Removing only the projection bias also changed the four initial scores by less than 0.001 and did not explain the overlap.

### 3.3 Multiwindow error diagnosis

The remaining original-TDNN multiwindow error was `member02_session02_006.wav`. It was 9.6626875 seconds long, with peak 0.518219, RMS 0.074246, and no detected clipping. Its RMS was within the batch range of 0.055543 to 0.080934. Low-energy frame counts were not treated as a speech activity detector.

| Model | member02 template | member03 template | Ranking |
|---|---:|---:|---|
| Original TDNN, multiple windows | 0.935215 | 0.936206 | Wrong by 0.000990 |
| CAM++, multiple windows | 0.783684 | 0.430326 | Correct |

The original TDNN ranked member03 higher in the first four windows and member02 higher in the final two. CAM++ ranked member02 higher in all six windows. The two original-TDNN multiwindow templates had cosine similarity 0.958081, showing limited separation in this embedding space. Similarity scales differ between models.

| Window in seconds | TDNN own minus other | CAM++ own minus other |
|---|---:|---:|
| 0 to 3 | -0.002750 | +0.246990 |
| 1.5 to 4.5 | -0.045149 | +0.230723 |
| 3 to 6 | -0.039926 | +0.322145 |
| 4.5 to 7.5 | -0.053739 | +0.292178 |
| 6 to 9 | +0.034865 | +0.399284 |
| 6.6626875 to 9.6626875 | +0.101055 | +0.379349 |

Removing window 2, 3, or 4 individually would reverse this TDNN ranking. This was a sensitivity check only; no windows were selected or discarded to repair the example. The evidence does not identify a particular noise source or prove that filtering would fix it.

### 3.4 What improved and what did not

- Multiple windows improved the original TDNN from 15/20 to 19/20 on independent session02 queries.
- The cosine classifier with 100 epochs achieved 136/136 public validation accuracy but only 16/20 multiwindow member rankings. Public classification accuracy alone did not predict member identification quality. Both classifier objective and epoch budget changed relative to the original baseline, so this comparison does not isolate either factor.
- Margin 0.1 at 100 epochs improved the matched cosine result from 16/20 to 18/20, but remained below the original TDNN's 19/20.
- Margin 0.2 at 100 epochs reached 20/20. Additional runs retained 20/20 but changed score gaps.
- More epochs did not consistently improve separation: margin 0.1 at 500 epochs had a smaller minimum gap than at 200; margin 0.2 at 1000 epochs had a smaller minimum gap than at 500.
- Margin 0.3 at 500 epochs slightly improved both minimum and mean gaps over margin 0.2 at 500 epochs. Compared with margin 0.3 at 200 epochs, it improved the minimum gap but reduced the mean gap.

## 4 Experimental results

### 4.1 Common evaluation protocol

Each identity uses ten session01 recordings for enrolment and ten session02 recordings for queries. Normalize each recording vector, average the ten vectors equally, and normalize the member template. For multiwindow processing, use three-second windows with a 1.5-second hop and append an end-aligned final window when needed. Normalize each window vector, average windows equally, and normalize the recording vector. Apply the same strategy to enrolment and queries, rebuilding templates for every checkpoint.

Rank templates by cosine similarity without an acceptance threshold. The score gap is the true-template score minus the strongest other-template score. Earlier member03 session01 checks included enrolment-included and leave-one-out testing; those same-session results are not used in the cross-session tables below. CAM++ controls use the same crop or window protocol as TDNN.

### 4.2 Centered crop and multiwindow comparison

| Model | Processing | member02 | member03 | Total |
|---|---|---:|---:|---:|
| Original TDNN, 10 epochs | Centered 3 seconds | 5/10 | 10/10 | 15/20 |
| Original TDNN, 10 epochs | Multiple windows | 9/10 | 10/10 | 19/20 |
| Cosine TDNN, 100 epochs | Centered 3 seconds | 5/10 | 8/10 | 13/20 |
| Cosine TDNN, 100 epochs | Multiple windows | 8/10 | 8/10 | 16/20 |
| AM-Softmax margin 0.1, 100 epochs | Centered 3 seconds | 8/10 | 9/10 | 17/20 |
| AM-Softmax margin 0.1, 100 epochs | Multiple windows | 8/10 | 10/10 | 18/20 |
| AM-Softmax margin 0.2, 100 epochs | Centered 3 seconds | 10/10 | 10/10 | 20/20 |
| AM-Softmax margin 0.2, 100 epochs | Multiple windows | 10/10 | 10/10 | 20/20 |
| Pretrained CAM++ | Centered 3 seconds | 10/10 | 10/10 | 20/20 |
| Pretrained CAM++ | Multiple windows | 10/10 | 10/10 | 20/20 |

### 4.3 Completed multiwindow training trials

| Training margin | Epoch budget | Selected epoch | member02 | member03 | Total | Minimum score gap | Mean score gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 200 | 152 | 10/10 | 10/10 | 20/20 | 0.068795 | 0.176637 |
| 0.1 | 500 | 421 | 10/10 | 10/10 | 20/20 | 0.041976 | 0.206349 |
| 0.2 | 200 | 177 | 10/10 | 10/10 | 20/20 | 0.148245 | 0.322675 |
| 0.2 | 500 | 430 | 10/10 | 10/10 | 20/20 | 0.198226 | 0.322118 |
| 0.2 | 1000 | 985 | 10/10 | 10/10 | 20/20 | 0.156002 | 0.338135 |
| 0.3 | 200 | 187 | 10/10 | 10/10 | 20/20 | 0.164892 | 0.401064 |
| 0.3 | 500 | 483 | 10/10 | 10/10 | 20/20 | 0.211960 | 0.328748 |

Margin 0.1 at 1000 epochs was skipped; it has no result. At 100 epochs, the minimum multiwindow gap was -0.028817 for margin 0.1 and 0.010212 for margin 0.2.

The margin 0.3 500-epoch run completed 38,000 updates in 976.4 seconds. Its selected epoch 483 achieved public validation loss 0.0000216654 and 136/136 correct. Checks confirmed finite loss, changed backbone weights, identical checkpoint reload outputs, and a non-smoke run. Checkpoint hashes and individual query summaries were verified.

At the matched 500-epoch budget, raising margin from 0.2 to 0.3 increased the minimum gap by 0.013735 and mean gap by 0.006631, with correct count unchanged at 20/20. By correct count followed by minimum gap, margin 0.3 at 500 epochs with multiwindow averaging is the strongest completed TDNN candidate in this two-member development comparison. Margin 0.3 at 200 epochs has a larger mean gap, so the ordering depends on the stated criterion.

### 4.4 Interference after adding member04

| Model | Two identities | Three identities | Queries predicted as member04 |
|---|---:|---:|---:|
| Original TDNN 10 epochs | 19/20 | 19/20 | 0/20 |
| Cosine TDNN 100 epochs | 16/20 | 16/20 | 0/20 |
| AM-Softmax margin 0.1 100 epochs | 18/20 | 18/20 | 0/20 |
| AM-Softmax margin 0.2 100 epochs | 20/20 | 20/20 | 0/20 |
| AM-Softmax margin 0.1 200 epochs | 20/20 | 20/20 | 0/20 |
| AM-Softmax margin 0.2 200 epochs | 20/20 | 20/20 | 0/20 |
| CAM++ | 20/20 | 20/20 | 0/20 |

Adding member04 did not change these member02/member03 rankings. This check used the listed existing models only; the later 500- and 1000-epoch models and margin 0.3 models are not covered by this three-identity table. It does not measure member04 cross-session recognition.

All twenty session02 queries have been reused for development choices. These results do not establish five-member performance or Unknown rejection. No security threshold was calibrated. Similarity scales differ between models, so raw gap sizes should not be treated as a common calibrated confidence scale.

## 5 Reproduction commands

Run these PowerShell examples from the project root after installing the recorded dependencies. Use the inference environment for CAM++ and the training environment for TDNN. Choose new output names for every independent run; the examples do not prescribe anyone's local storage layout.

### 5.1 Check the prepared data

```powershell
$python = ".\.venv-train\Scripts\python.exe"
Get-Content .\data\manifests\small_model\summary.json
```

The expected split is 28 speakers with 1210 training, 136 validation, and 136 test recordings. Keep the existing split fixed for a controlled comparison.

### 5.2 Train one isolated experiment

Set the epoch budget and margin for the experiment. The output value is selected by the person running the command.

```powershell
$epochs = 200
$margin = 0.2
$runOutput = Read-Host "Choose a new output directory for this run"

& $python .\train_small.py train --device cuda --epochs $epochs --batch-size 16 --seed 3601 --lr 0.001 --cosine-classifier --margin $margin --output $runOutput
```

| Experiment family | Epoch budgets used | Classifier options |
|---|---|---|
| Original linear baseline | 10 | Omit both `--cosine-classifier` and `--margin` |
| Normalized cosine baseline | 100 | Use `--cosine-classifier --margin 0` |
| Additive cosine margin | 100, 200, 500, and the authorized 1000-epoch trial | Use `--cosine-classifier --margin $margin`; tested values are 0.1, 0.2, and 0.3 |

The result tables identify which combinations actually ran. This table is not an instruction to run the entire Cartesian product. The 1000-epoch margin 0.1 job was skipped under the recorded condition. These commands initialize new weights; they do not resume an existing checkpoint. For a CPU-only machine, replace `--device cuda` with `--device cpu` and record that change.

### 5.3 Check the selected model

```powershell
Get-Content (Join-Path $runOutput "verification.json")
$checkpoint = Join-Path $runOutput "best.pt"
& $python .\train_small.py evaluate --device cuda --checkpoint $checkpoint --split valid
```

Check that the run is not smoke-only, loss is finite, backbone weights changed, and checkpoint reload equality passed. `best.pt` is selected by minimum public validation loss; `last.pt` is the final epoch. The evaluation command above uses validation only and writes its validation result with the chosen run. Do not substitute `--split test` while choosing experiments.

### 5.4 Extract and inspect a recording

```powershell
$wav = Read-Host "Choose a mono 16 kHz WAV recording"
$vectorOutput = Read-Host "Choose an output filename for the embedding"
& $python .\train_small.py embed --device cuda --checkpoint $checkpoint --wav $wav --vector-output $vectorOutput

$inferencePython = ".\.venv\Scripts\python.exe"
& $inferencePython .\voiceid.py inspect $wav
```

The existing TDNN `embed` command uses the centered three seconds. It does not perform multiwindow aggregation or member identification by itself.

### 5.5 Reproduce the multiwindow comparison

Use the same algorithm for enrolment and queries:

1. Read mono 16 kHz PCM audio. Use three-second windows with a 1.5-second hop, appending the end-aligned final window if needed.
2. Extract a 64-dimensional TDNN embedding per window in evaluation mode with gradients disabled.
3. Normalize each window embedding, average windows equally, and normalize the recording vector.
4. Average the ten session01 recording vectors equally for each member, then normalize that member's template.
5. Query using session02 recording vectors. Rank templates by dot product. Record the predicted identity and the true-template score minus the strongest other-template score.

The same recording and aggregation rules apply to the CAM++ control, using its own 192-dimensional embeddings. Rebuild templates for every checkpoint and processing strategy. The batch comparisons described in this record were performed by separate diagnostic scripts; there is no `--multiwindow` option in the current `train_small.py` CLI. A portable batch runner must implement these rules rather than invent a CLI option or reuse templates from another model.

### 5.6 CAM++ registration and query commands

For the existing full-recording CAM++ CLI, supply session01 files for registration and independent session02 files for queries:

```powershell
$database = Read-Host "Choose the CAM++ template database"
$enrolWavs = (Read-Host "Enter session01 WAV paths separated by semicolons").Split(";")
& $inferencePython .\voiceid.py enrol member04 $enrolWavs --db $database

$threshold = Read-Host "Enter the threshold fixed using validation data"
& $inferencePython .\voiceid.py query $wav --threshold $threshold --db $database
```

These CLI commands process full recordings and are not the matched three-second or multiwindow control used in the comparison tables. Do not treat the earlier demonstration value 0.5 as a calibrated security threshold. Enrolment changes templates, not network weights, and re-enrolling an identity replaces its template.
