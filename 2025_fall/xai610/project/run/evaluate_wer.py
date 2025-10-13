from jiwer import wer

# Suppose your data is in a text file like "results.txt" with lines:
# REF: ...
# HYP: ...
file_path = "ttt"

refs = []
hyps = []

with open(file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]  # remove empty lines

for i in range(0, len(lines), 2):  # every two lines: REF and HYP
    ref_line = lines[i]
    hyp_line = lines[i + 1]

    # Extract text after "REF:" / "HYP:"
    ref_text = ref_line[len("REF:"):].strip()
    hyp_text = hyp_line[len("HYP:"):].strip()

    refs.append(ref_text)
    hyps.append(hyp_text)

# Compute WER
test_wer = wer(refs, hyps)
print(f"WER: {test_wer:.4f}")

