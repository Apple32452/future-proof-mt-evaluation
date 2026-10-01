"""
Generate WMT25 translations with a frozen 2024-era Aya-23 model.

Purpose:
    Build a genuinely historical difficulty signal for the
    temporal-transfer experiment.

Model:
    mlx-community/aya-23-8B-4bit

Language pairs:
    cs-uk_UA
    en-cs_CZ
    en-ja_JP
    en-ru_RU
    en-uk_UA
    en-zh_CN

Input:
    results/wmt25_subset2evaluate_all/
        <language_pair>_subset2evaluate.jsonl

Output:
    results/temporal_aya_wmt25/
        <language_pair>_aya23_translations.jsonl

The script checkpoints after EVERY sentence.
If interrupted, simply rerun it.
"""

import json
from pathlib import Path

from mlx_lm import load, generate


# ============================================================
# Configuration
# ============================================================

MODEL = "mlx-community/aya-23-8B-4bit"

LANGUAGE_PAIRS = {
    "cs-uk_UA": (
        "Czech",
        "Ukrainian",
    ),
    "en-cs_CZ": (
        "English",
        "Czech",
    ),
    "en-ja_JP": (
        "English",
        "Japanese",
    ),
    "en-ru_RU": (
        "English",
        "Russian",
    ),
    "en-uk_UA": (
        "English",
        "Ukrainian",
    ),
    "en-zh_CN": (
        "English",
        "Simplified Chinese",
    ),
}

MAX_TOKENS = 256


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "wmt25_subset2evaluate_all"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "temporal_aya_wmt25"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load model once
# ============================================================

print("=" * 100)
print("TEMPORAL AYA-23 → WMT25 TRANSLATION")
print("=" * 100)

print(f"\nLoading model: {MODEL}")

model, tokenizer = load(
    MODEL
)

print("Model loaded successfully.")


# ============================================================
# Translation helper
# ============================================================

def translate(
    source_text,
    source_language,
    target_language,
):

    messages = [
        {
            "role": "user",
            "content": (
                f"Translate the following text from "
                f"{source_language} to {target_language}. "
                f"Return only the translation, with no "
                f"explanation or additional text.\n\n"
                f"{source_text}"
            ),
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        verbose=False,
    )

    return output.strip()


# ============================================================
# Process languages
# ============================================================

for (
    language_pair,
    (
        source_language,
        target_language,
    ),
) in LANGUAGE_PAIRS.items():

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"{language_pair}: "
        f"{source_language} → {target_language}"
    )

    print(
        "=" * 100
    )


    input_file = (
        DATA_DIR
        / (
            f"{language_pair}"
            "_subset2evaluate.jsonl"
        )
    )


    output_file = (
        OUTPUT_DIR
        / (
            f"{language_pair}"
            "_aya23_translations.jsonl"
        )
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Missing input file:\n"
            f"{input_file}"
        )


    # --------------------------------------------------------
    # Load WMT25 items
    # --------------------------------------------------------

    items = []

    with open(
        input_file,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            items.append(
                json.loads(line)
            )


    print(
        f"Input sentences: "
        f"{len(items)}"
    )


    # --------------------------------------------------------
    # Load previous checkpoint
    # --------------------------------------------------------

    completed = {}


    if output_file.exists():

        with open(
            output_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                row = json.loads(line)

                completed[
                    int(
                        row[
                            "i"
                        ]
                    )
                ] = row


    print(
        f"Already completed: "
        f"{len(completed)}"
    )


    # --------------------------------------------------------
    # Generate missing translations
    # --------------------------------------------------------

    for position, item in enumerate(
        items,
        start=1,
    ):

        item_id = int(
            item["i"]
        )


        if item_id in completed:

            continue


        source_text = (
            item["src"]
        )


        try:

            translation = translate(
                source_text,
                source_language,
                target_language,
            )


        except Exception as exc:

            print(
                f"\nERROR at "
                f"{language_pair} "
                f"item {item_id}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            raise


        output_row = {
            "i":
                item_id,

            "language_pair":
                language_pair,

            "source_language":
                source_language,

            "target_language":
                target_language,

            "src":
                source_text,

            "ref":
                item.get(
                    "ref",
                    "",
                ),

            "aya23_translation":
                translation,

            "model":
                MODEL,
        }


        # Append immediately:
        # one sentence = one checkpoint.
        with open(
            output_file,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                )
                + "\n"
            )


        completed[
            item_id
        ] = output_row


        print(
            f"[{language_pair}] "
            f"{len(completed):3d}/"
            f"{len(items):3d} | "
            f"id={item_id}"
        )


    print(
        f"\n{language_pair} COMPLETE: "
        f"{len(completed)}/"
        f"{len(items)}"
    )


# ============================================================
# Done
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "ALL TEMPORAL AYA TRANSLATIONS COMPLETE"
)

print(
    "=" * 100
)

print(
    f"\nSaved under:\n"
    f"{OUTPUT_DIR}"
)
