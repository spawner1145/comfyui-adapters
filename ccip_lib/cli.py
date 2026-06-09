import argparse
import sys
from pathlib import Path

try:
    from .ccip import (
        ccip_extract_feature,
        ccip_batch_extract_features,
        ccip_difference,
        ccip_same,
    )
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ccip_lib.ccip import (
        ccip_extract_feature,
        ccip_batch_extract_features,
        ccip_difference,
        ccip_same,
    )


def _parse_args():
    p = argparse.ArgumentParser(prog='ccip-cli', description='Simple CCIP inference CLI (local model folder supported)')
    sub = p.add_subparsers(dest='cmd')

    ex = sub.add_parser('extract', help='Extract feature of a single image')
    ex.add_argument('image')
    ex.add_argument('--model-dir', '-m', help='Local model folder (containing model_feat.onnx etc.)')
    ex.add_argument('--size', type=int, default=384)
    ex.add_argument('--out', '-o', help='Save feature to .npy file')

    diff = sub.add_parser('diff', help='Compute difference between two images')
    diff.add_argument('image1')
    diff.add_argument('image2')
    diff.add_argument('--model-dir', '-m')
    diff.add_argument('--size', type=int, default=384)

    same = sub.add_parser('same', help='Decide whether two images are same character')
    same.add_argument('image1')
    same.add_argument('image2')
    same.add_argument('--model-dir', '-m')
    same.add_argument('--size', type=int, default=384)
    same.add_argument('--threshold', type=float)

    return p.parse_args()


def main(argv=None):
    args = _parse_args() if argv is None else _parse_args()

    model = args.model_dir if hasattr(args, 'model_dir') and args.model_dir else None

    if args.cmd == 'extract':
        feat = ccip_extract_feature(args.image, size=args.size, model=model or None)
        print(feat.shape, feat.dtype)
        if args.out:
            import numpy as _np
            _np.save(args.out, feat)

    elif args.cmd == 'diff':
        diff = ccip_difference(args.image1, args.image2, size=args.size, model=model or None)
        print(diff)

    elif args.cmd == 'same':
        res = ccip_same(args.image1, args.image2, threshold=getattr(args, 'threshold', None), size=args.size, model=model or None)
        print(res)

    else:
        print('No command. Use -h for help.')


if __name__ == '__main__':
    main()
