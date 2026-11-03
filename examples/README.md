# Examples

Eight samples from MMSI (speaking target `T`, mentioned player `N`, pronoun coreference `P`) and TVQA+ (`QA`).
Each folder holds the 8 uniformly sampled 640x360 frames that are fed to the model and the source clip
`clip.mp4` (MMSI: the 3-second window of the original video; TVQA+: the 3-fps frames of the question's time
span); `samples.json` holds the question, the answer, the speaker boxes and the speaker spans.

```bash
python run_zeroshot.py --model qwen2_5_vl --data examples/samples.json --output outputs/qwen_base.jsonl --no-bias
python run_zeroshot.py --model qwen2_5_vl --data examples/samples.json --output outputs/qwen_ours.jsonl
python evaluate.py outputs/qwen_base.jsonl outputs/qwen_ours.jsonl
```

The frames come from the MMSI (Werewolf YouTube videos) and TVQA+ (The Big Bang Theory) benchmarks and are
included for demonstration only; please obtain the full datasets from their authors.
