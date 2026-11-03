"""Command-line entry point. Errors are explicit and return a non-zero exit."""
import argparse
import csv
import json
from pathlib import Path
import sys
from PIL import Image
from .contracts import Spacing
from .dataset import read_manifest, import_triplets
from .io import load_image, load_mask
from .pipeline import analyze, demo_result, predict, write_result


def parser():
    p = argparse.ArgumentParser(description="Kidney segmentation research workbench; not a diagnostic system")
    commands = p.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Export a labelled synthetic engineering fixture")
    demo.add_argument("--output", type=Path, default=Path("results/demo"))
    demo.add_argument("--calibrated", action="store_true")
    a = commands.add_parser("analyze", help="Measure a supplied label mask and export an overlay")
    a.add_argument("--image", type=Path, required=True)
    a.add_argument("--mask", type=Path, required=True)
    a.add_argument("--reference", type=Path)
    a.add_argument("--spacing", type=float, nargs=2, metavar=("ROW_MM", "COLUMN_MM"))
    a.add_argument("--depth-mm", type=float)
    a.add_argument("--label-schema", choices=["corrected-five", "source-four"], default="corrected-five")
    a.add_argument("--output", type=Path, required=True)
    t = commands.add_parser("train", help="Train from an explicit patient-separated data manifest")
    t.add_argument("--manifest", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--loss", choices=["focal", "cross-entropy"], default="focal")
    t.add_argument("--pretrained-backbone", help="Explicit Hugging Face model ID or local backbone; may download weights")
    pred = commands.add_parser("predict", help="Infer using a trained safetensors model manifest")
    pred.add_argument("--model", type=Path, required=True)
    pred.add_argument("--image", type=Path, required=True)
    pred.add_argument("--spacing", type=float, nargs=2)
    pred.add_argument("--output", type=Path, required=True)
    batch = commands.add_parser("batch", help="Analyze image/mask pairs listed in a data manifest")
    batch.add_argument("--manifest", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check-data", help="Validate labels, pairs and patient/image split separation")
    check.add_argument("--manifest", type=Path, required=True)
    imp = commands.add_parser("import-triplets", help="Convert original capsule/region masks without losing region IDs")
    for key in ["images", "capsules", "regions", "output"]:
        imp.add_argument("--" + key, type=Path, required=True)
    web = commands.add_parser("serve", help="Open the local research workbench")
    web.add_argument("--port", type=int, default=8782)
    web.add_argument("--models-dir", type=Path, default=Path("models"))
    models = commands.add_parser("models", help="List supplied checkpoints, available files and explicit input profiles")
    models.add_argument("--models-dir", type=Path, default=Path("models"))
    fetch = commands.add_parser("fetch-models", help="Verify/download private release assets using existing gh authentication")
    fetch.add_argument("--models-dir", type=Path, default=Path("models"))
    fetch.add_argument("--include-originals", action="store_true")
    convert = commands.add_parser("import-models", help="Convert the three exact supplied originals without unrestricted pickle")
    convert.add_argument("--source-dir", type=Path, required=True)
    convert.add_argument("--models-dir", type=Path, required=True)
    run = commands.add_parser("run-model", help="Run a supplied checkpoint with an explicitly selected research input profile")
    run.add_argument("--id", required=True)
    run.add_argument("--image", type=Path, required=True)
    run.add_argument("--profile", required=True)
    run.add_argument("--models-dir", type=Path, default=Path("models"))
    run.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            result = demo_result(calibrated=args.calibrated)
            write_result(result, args.output, "engineering-fixture")
            print(json.dumps(result["report"], indent=2))
        elif args.command == "analyze":
            result = analyze(load_image(args.image), load_mask(args.mask),
                             spacing=Spacing(*args.spacing) if args.spacing else None,
                             reference=load_mask(args.reference) if args.reference else None,
                             depth_mm=args.depth_mm, label_schema_id=args.label_schema)
            write_result(result, args.output, protected=[args.image, args.mask, args.reference])
            print(json.dumps(result["report"], indent=2))
        elif args.command == "predict":
            result = predict(args.model, args.image, spacing=Spacing(*args.spacing) if args.spacing else None)
            write_result(result, args.output, protected=[args.image, args.model])
            print(json.dumps(result["report"], indent=2))
        elif args.command == "train":
            from .training import train
            print(train(args.manifest, args.output, epochs=args.epochs, batch_size=args.batch_size,
                        pretrained_backbone=args.pretrained_backbone, loss_name=args.loss))
        elif args.command == "check-data":
            samples = read_manifest(args.manifest)
            print(json.dumps({"samples": len(samples), "patients": len({s.patient_id for s in samples}),
                              "splits": {s: sum(x.split == s for x in samples) for s in ["train", "val", "test"]}}))
        elif args.command == "import-triplets":
            print(json.dumps({"imported": import_triplets(args.images, args.capsules, args.regions, args.output),
                              "next_step": "Create manifest.csv with explicit patient-separated train/val/test assignments"}))
        elif args.command == "batch":
            samples = read_manifest(args.manifest)
            reports = []
            for number, sample in enumerate(samples):
                result = analyze(load_image(sample.image), load_mask(sample.mask))
                write_result(result, args.output, f"{number + 1:04d}-{sample.image.stem}",
                             protected=[p for s in samples for p in [s.image, s.mask]])
                reports.append(result["report"])
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "batch.json").write_text(json.dumps(reports, indent=2) + "\n")
            with (args.output / "summary.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample", "foreground_pixels", "coverage_fraction"])
                for sample, report in zip(samples, reports):
                    writer.writerow([sample.image.name, report["measurements"]["foreground_pixels"], report["measurements"]["coverage_fraction"]])
            print(f"Analyzed {len(samples)} samples; exports saved to {args.output}")
        elif args.command == "serve":
            from .server import serve
            serve(port=args.port, models_directory=args.models_dir)
        elif args.command == "models":
            from .supplied import model_status
            print(json.dumps(model_status(args.models_dir), indent=2))
        elif args.command == "fetch-models":
            from .supplied import fetch_models
            print(json.dumps({"verified_files": fetch_models(args.models_dir, include_originals=args.include_originals)}, indent=2))
        elif args.command == "import-models":
            from .supplied import registry, convert_original
            for spec in registry()["models"]:
                print(json.dumps(convert_original(spec["id"], args.source_dir / spec["original"]["name"], args.models_dir)))
        elif args.command == "run-model":
            from .supplied import run_model
            result = run_model(args.id, args.image.read_bytes(), args.models_dir, profile_id=args.profile)
            write_result(result, args.output, "model-output", protected=[args.image])
            print(json.dumps(result["report"], indent=2))
        return 0
    except (ValueError, FileNotFoundError, OSError, ImportError, Image.DecompressionBombError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
