#!/usr/bin/env python3
"""Prepare raw ID label rasters for Semantic A-TDOM training."""

from argparse import ArgumentParser

from semantic_atdom.semantics.id_label_adapter import prepare_id_label_dataset


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--adapter-spec", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = prepare_id_label_dataset(
        label_root=args.label_root,
        image_root=args.image_root,
        adapter_spec_path=args.adapter_spec,
        output_root=args.output_root,
    )
    print("Prepared {} semantic observations in {}".format(
        result.image_count, result.dataset_root
    ))
    print("Adapter spec SHA256: {}".format(result.adapter_spec_sha256))
    print("Manifest: {}".format(result.manifest_path))


if __name__ == "__main__":
    main()
