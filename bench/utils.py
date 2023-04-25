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


def prepare_clef_2020_corpus(
        file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        original_lines = f_p.readlines()

    lines = []

    # Add missing newline after header
    lines.append(original_lines[0])

    for line in original_lines[1:]:
        if line.startswith(" \t"):
            # Workaround for empty tokens
            continue

        line = line.strip()

        # Add "real" document marker
        if add_document_separator and line.startswith(document_separator):
            lines.append("-DOCSTART- O")
            lines.append("")

        lines.append(line)

        if eos_marker in line:
            lines.append("")

    # Now here comes the de-hyphenation part ;)
    word_seperator = "¬"

    for index, line in enumerate(lines):
        if line.startswith("#"):
            continue

        if line.startswith(word_seperator):
            continue

        if not line:
            continue

        prev_line = lines[index - 1]

        prev_prev_line = lines[index - 2]

        if not prev_line.startswith(word_seperator):
            continue

        # Example:
        # Po  <- prev_prev_line
        # ¬   <- prev_line
        # len <- current_line
        #
        # will be de-hyphenated to:
        #
        # Polen Dehyphenated-3
        # # ¬
        # # len
        suffix = line.split("\t")[0]

        prev_prev_line_splitted = lines[index - 2].split("\t")
        prev_prev_line_splitted[0] += suffix

        prev_line_splitted = lines[index - 1].split("\t")
        prev_line_splitted[0] = "#" + prev_line_splitted[0]
        prev_line_splitted[-1] += "|Commented"

        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "#" + current_line_splitted[0]
        current_line_splitted[-1] += "|Commented"

        # Add some meta information about suffix length
        # Later, it is possible to re-construct original token and suffix
        prev_prev_line_splitted[9] += f"|Dehyphenated-{len(suffix)}"

        lines[index - 2] = "\t".join(prev_prev_line_splitted)
        lines[index - 1] = "\t".join(prev_line_splitted)
        lines[index] = "\t".join(current_line_splitted)

    # Post-Processing I
    for index, line in enumerate(lines):
        if not line:
            continue

        if not line.startswith(word_seperator):
            continue

        # oh noooo
        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "#" + current_line_splitted[0]

        current_line_splitted[-1] += "|Commented"

        lines[index] = "\t".join(current_line_splitted)

    # Post-Processing II
    # Beautify: _|Commented –> Commented
    for index, line in enumerate(lines):
        if not line:
            continue

        if not line.startswith("#"):
            continue

        current_line_splitted = line.split("\t")

        if current_line_splitted[-1] == "_|Commented":
            current_line_splitted[-1] = "Commented"
            lines[index] = "\t".join(current_line_splitted)

    # Finally, save it!
    with open(file_out, "wt") as f_out:
        for line in lines:
            f_out.write(line + "\n")


def prepare_newseye_fi_sv_corpus(
    file_in: Path, file_out: Path, eos_marker: str, document_separator: str, add_document_separator: bool
):
    with open(file_in, "rt") as f_p:
        original_lines = f_p.readlines()

    lines = []

    # Add missing newline after header
    lines.append(original_lines[0])

    for line in original_lines[1:]:
        if line.startswith(" \t"):
            # Workaround for empty tokens
            continue

        line = line.strip()

        # Add "real" document marker
        if add_document_separator and line.startswith(document_separator):
            lines.append("-DOCSTART- O")
            lines.append("")

        lines.append(line)

        if eos_marker in line:
            lines.append("")

    # Now here comes the de-hyphenation part
    # And we want to avoid matching "-DOCSTART-" lines here, so append a tab
    word_seperator = "-\t"

    for index, line in enumerate(lines):
        if line.startswith("#"):
            continue

        if line.startswith(word_seperator):
            continue

        if not line:
            continue

        prev_line = lines[index - 1]

        prev_prev_line = lines[index - 2]

        if not prev_line.startswith(word_seperator):
            continue

        # Example:
        # Po  NoSpaceAfter <- prev_prev_line
        # -   <- prev_line
        # len <- current_line
        #
        # will be de-hyphenated to:
        #
        # Polen Dehyphenated-3
        # # -
        # # len
        #
        # It is really important, that "NoSpaceAfter" in the previous
        # line before hyphenation character! Otherwise, it is no real
        # hyphenation!

        if not "NoSpaceAfter" in prev_line:
            continue

        if not prev_prev_line:
            continue

        suffix = line.split("\t")[0]

        prev_prev_line_splitted = lines[index - 2].split("\t")
        prev_prev_line_splitted[0] += suffix

        prev_line_splitted = lines[index - 1].split("\t")
        prev_line_splitted[0] = "# " + prev_line_splitted[0]
        prev_line_splitted[-1] += "|Commented"

        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "# " + current_line_splitted[0]
        current_line_splitted[-1] += "|Commented"

        # Add some meta information about suffix length
        # Later, it is possible to re-construct original token and suffix
        prev_prev_line_splitted[9] += f"|Dehyphenated-{len(suffix)}"

        lines[index - 2] = "\t".join(prev_prev_line_splitted)
        lines[index - 1] = "\t".join(prev_line_splitted)
        lines[index]     = "\t".join(current_line_splitted)

    # Post-Processing I
    for index, line in enumerate(lines):
        if not line:
            continue

        if not line.startswith(word_seperator):
            continue

        # oh noooo
        current_line_splitted = line.split("\t")
        current_line_splitted[0] = "# " + current_line_splitted[0]

        current_line_splitted[-1] += "|Commented"

        lines[index] = "\t".join(current_line_splitted)

    # Post-Processing II
    # Beautify: _|Commented –> Commented
    for index, line in enumerate(lines):
        if not line:
            continue

        if not line.startswith("#"):
            continue

        current_line_splitted = line.split("\t")

        if current_line_splitted[-1] == "_|Commented":
            current_line_splitted[-1] = "Commented"
            lines[index] = "\t".join(current_line_splitted)

    # Finally, save it!
    with open(file_out, "wt") as f_out:
        for line in lines:
            f_out.write(line + "\n")
