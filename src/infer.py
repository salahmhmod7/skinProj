"""CLI inference: python -m src.infer --image <path> [--model <ckpt>] [--no-threshold]."""
import argparse
import json
import os

from .model import MEL_THRESHOLD, MODEL_PATH, load_model, predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="HAM10000 skin-lesion inference (ConvNeXt-Base)")
    parser.add_argument("--image", required=True, help="Path to a dermoscopy image (.jpg/.png)")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", MODEL_PATH))
    parser.add_argument("--no-threshold", action="store_true",
                        help="Disable melanoma-threshold rule, use plain argmax")
    parser.add_argument("--mel-threshold", type=float, default=MEL_THRESHOLD)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    print("Loading model...")
    model = load_model(args.model)
    print("Analyzing image...")
    res = predict_image(args.image, model,
                        apply_threshold=not args.no_threshold,
                        mel_threshold=args.mel_threshold)

    print("=" * 55)
    print(f" Diagnosis : {res['pred_label'].upper()}")
    print(f" Meaning   : {res['description']}")
    print(f" Confidence: {res['confidence']:.2f}%")
    print(f" Mel prob  : {res['mel_probability']:.2f}% (threshold {res['mel_threshold']:.0f}%)")
    print("=" * 55)
    print("\nAll classes:")
    for name, prob in res["all_probabilities"].items():
        print(f"  - {name:5s}: {prob:6.2f}%")
    print("\nJSON:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
