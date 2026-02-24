import argparse
import json
import os
import struct
import sys


def load_header(path: str):
    with open(path, "rb") as f:
        header_len_raw = f.read(8)
        if len(header_len_raw) != 8:
            raise ValueError("invalid safetensors: missing header length")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError("invalid safetensors: incomplete header")
    header = json.loads(header_raw.decode("utf-8"))
    return header_len, header


def build_new_header(old_header: dict, remove_key: str):
    if remove_key not in old_header:
        raise KeyError(f"tensor key not found: {remove_key}")

    remove_entry = old_header[remove_key]
    if not isinstance(remove_entry, dict):
        raise ValueError(f"invalid tensor entry for key: {remove_key}")

    remove_start, remove_end = remove_entry["data_offsets"]
    remove_size = int(remove_end) - int(remove_start)
    if remove_size <= 0:
        raise ValueError("invalid remove range")

    new_header = {}
    for key, value in old_header.items():
        if key == remove_key:
            continue
        if key == "__metadata__":
            new_header[key] = value
            continue
        if not isinstance(value, dict):
            new_header[key] = value
            continue

        start, end = value["data_offsets"]
        start = int(start)
        end = int(end)
        if end <= remove_start:
            new_offsets = [start, end]
        elif start >= remove_end:
            new_offsets = [start - remove_size, end - remove_size]
        else:
            raise ValueError(
                f"tensor overlaps removed region: {key} offsets={value['data_offsets']}"
            )

        new_value = dict(value)
        new_value["data_offsets"] = new_offsets
        new_header[key] = new_value

    return new_header, int(remove_start), int(remove_end), int(remove_size)


def write_deduped_file(
    src_path: str, dst_path: str, old_header_len: int, new_header: dict, remove_start: int, remove_end: int
):
    new_header_raw = json.dumps(new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    new_header_len = len(new_header_raw)

    data_offset = 8 + old_header_len
    copy_chunk_size = 8 * 1024 * 1024

    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        dst.write(struct.pack("<Q", new_header_len))
        dst.write(new_header_raw)

        src.seek(data_offset)
        remaining = remove_start
        while remaining > 0:
            chunk = src.read(min(copy_chunk_size, remaining))
            if not chunk:
                raise ValueError("unexpected EOF while copying first segment")
            dst.write(chunk)
            remaining -= len(chunk)

        src.seek(data_offset + remove_end)
        while True:
            chunk = src.read(copy_chunk_size)
            if not chunk:
                break
            dst.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove duplicate lm_head tensor from a safetensors file."
    )
    parser.add_argument("--input", default="model.safetensors", help="Source safetensors file")
    parser.add_argument(
        "--output",
        default="model_dedup.safetensors",
        help="Output safetensors file without lm_head",
    )
    parser.add_argument(
        "--remove-key",
        default="lm_head.weight",
        help="Tensor key to remove (default: lm_head.weight)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"error: input file not found: {args.input}")
        return 1
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("error: input and output paths must be different")
        return 1

    old_header_len, old_header = load_header(args.input)
    new_header, remove_start, remove_end, remove_size = build_new_header(
        old_header, args.remove_key
    )

    print("input:", args.input)
    print("output:", args.output)
    print("remove key:", args.remove_key)
    print("remove range:", [remove_start, remove_end], f"({round(remove_size / 1024 / 1024, 2)} MB)")

    write_deduped_file(
        args.input, args.output, old_header_len, new_header, remove_start, remove_end
    )

    old_size = os.path.getsize(args.input)
    new_size = os.path.getsize(args.output)
    print("old size MB:", round(old_size / 1024 / 1024, 2))
    print("new size MB:", round(new_size / 1024 / 1024, 2))
    print("saved MB:", round((old_size - new_size) / 1024 / 1024, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
