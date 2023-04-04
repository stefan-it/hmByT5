from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

from pathlib import Path

from typing import List


def prepare_ajmc_corpus(
    file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        lines = f_p.readlines()

    with open(file_out, "wt") as f_out:
        # Add missing newline after header
        f_out.write(lines[0] + "\n")

        for line in lines[1:]:
            if line.startswith(" \t"):
                # Workaround for empty tokens
                continue

            line = line.strip()

            # HIPE-2022 late pre-submission fix:
            # Our hmBERT model has never seen Fraktur, so we replace long s
            line = line.replace("ſ", "s")

            # Add "real" document marker
            if add_document_separator and line.startswith(document_separator):
                f_out.write("-DOCSTART- O\n\n")

            f_out.write(line + "\n")

            if eos_marker in line:
                    f_out.write("\n")

    print("Special preprocessing for AJMC has finished!")
