"""Zero-shot evaluation of an MLLM with / without the social-aware attention bias.

Example::

    python run_zeroshot.py --model qwen2_5_vl --data examples/samples.json --output preds.jsonl
    python run_zeroshot.py --model internvl3 --data examples/samples.json --no-bias   # plain baseline
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from social_attention import load_model  # noqa: E402
from social_attention.data import load_frames, load_samples, normalize_answer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["qwen2_5_vl", "internvl3"])
    p.add_argument("--model-path", default=None, help="HF repo id or local path (default: the 7B/8B instruct model)")
    p.add_argument("--data", required=True, help="JSON file with samples (see social_attention/data.py)")
    p.add_argument("--output", default=None, help="where to write predictions (jsonl)")
    p.add_argument("--no-bias", action="store_true", help="run the unmodified model")
    p.add_argument("--alpha", type=float, default=None, help="bias strength alpha (default: model-specific)")
    p.add_argument("--threshold", type=float, default=None, help="active-head threshold lambda (default: model-specific)")
    p.add_argument("--layer-min", type=int, default=None, help="first biased layer (default: 10)")
    p.add_argument("--layer-max", type=int, default=None, help="last biased layer (default: 19)")
    p.add_argument("--visual-region", choices=["all", "boxes"], default=None, help="V_all for head selection / adaptive max (default: boxes)")
    p.add_argument("--num-frames", type=int, default=8, help="frames loaded per sample (uniformly sub-sampled)")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--tasks", nargs="*", default=None, help="only evaluate samples whose task is in this list")
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug", action="store_true", help="print the text covered by each speaker's token spans")
    p.add_argument(
        "--head-stats",
        action="store_true",
        help="report the ratio of active heads per layer (use it to pick --threshold so that ~10-20%% are active)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    kwargs = {"device": args.device}
    if args.model_path:
        kwargs["model_name_or_path"] = args.model_path
    model = load_model(args.model, **kwargs)
    cfg = model.bias_config  # model-specific zero-shot defaults, overridden by the command line
    for name, value in [("alpha", args.alpha), ("head_threshold", args.threshold), ("layer_min", args.layer_min), ("layer_max", args.layer_max), ("visual_region", args.visual_region)]:
        if value is not None:
            setattr(cfg, name, value)
    cfg.record_stats = args.head_stats
    print(f"bias config: {cfg}")

    samples = load_samples(args.data)
    if args.tasks:
        samples = [s for s in samples if s.get("task") in args.tasks]
    if args.max_samples:
        samples = samples[: args.max_samples]

    out = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        out = open(args.output, "w", encoding="utf-8")
    correct, total = defaultdict(int), defaultdict(int)
    layer_active, layer_mean = defaultdict(list), defaultdict(list)
    t0 = time.time()
    for i, s in enumerate(samples):
        frames = load_frames(s["frames"], args.num_frames)
        pred, dbg = model.generate(
            frames,
            s["question"],
            s.get("speaker_boxes"),
            s.get("speaker_spans"),
            use_bias=not args.no_bias,
            max_new_tokens=args.max_new_tokens,
            return_debug=True,
        )
        rec = {"id": s.get("id", i), "task": s.get("task"), "predict": pred, "label": s.get("answer")}
        for st in dbg.get("head_stats", []):
            layer_active[st["layer"]].append(st["active_ratio"])
            layer_mean[st["layer"]].append(st["mean_cross_attn"])
        if args.debug and "token_spans" in dbg:
            rec["token_spans"] = dbg["token_spans"]
            print(json.dumps(dbg["token_spans"], ensure_ascii=False, indent=1))
        if out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        if s.get("answer") is not None:
            task = s.get("task", "all")
            total[task] += 1
            correct[task] += int(normalize_answer(pred) == normalize_answer(s["answer"]))
        if (i + 1) % 20 == 0 or i + 1 == len(samples):
            acc = {t: f"{correct[t] / total[t]:.1%}" for t in total}
            print(f"[{i + 1}/{len(samples)}] {time.time() - t0:.0f}s acc={acc}", flush=True)
    if out:
        out.close()
    if layer_active:
        print(f"Active heads per layer at threshold {cfg.head_threshold:g} (mean cross-modal attention x1e-5):")
        for layer in sorted(layer_active):
            n = len(layer_active[layer])
            print(f"  layer {layer:2d}: {sum(layer_active[layer]) / n:.1%} active, attn {sum(layer_mean[layer]) / n * 1e5:.2f}")
        all_ratios = [r for v in layer_active.values() for r in v]
        print(f"  overall: {sum(all_ratios) / len(all_ratios):.1%} active")
    if total:
        print("Accuracy by task:")
        for t in sorted(total):
            print(f"  {t}: {correct[t] / total[t]:.2%} ({correct[t]}/{total[t]})")
        print(f"  overall: {sum(correct.values()) / sum(total.values()):.2%}")


if __name__ == "__main__":
    main()
