#!/usr/bin/env bash
# Downloads AMI manual annotations, the Mix-Headset audio of the selected
# meetings, and the BUT reference RTTMs for the test set.
#
# Usage: bash download_ami.sh <data_dir> [num_train_meetings]
#   Disk: about 60 MB per meeting (60 train + 18 dev + 16 test ~ 5.6 GB).
set -euo pipefail
DATA=${1:-data}
NUM_TRAIN=${2:-60}
AMI=https://groups.inf.ed.ac.uk/ami
SETUP=https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main
mkdir -p "$DATA/wav" "$DATA/ref_rttm" "$DATA/ref_uem"

# 1. Annotations (words, speakers, summaries): 23 MB
if [ ! -d "$DATA/ami_annotations/words" ]; then
  wget -q -O "$DATA/ann.zip" "$AMI/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
  unzip -q "$DATA/ann.zip" -d "$DATA/ami_annotations" && rm "$DATA/ann.zip"
fi

# 2. Meeting lists (Full-corpus-ASR split). Train: a fixed random subset.
for s in train dev test; do wget -q -O "$DATA/$s.meetings.txt" "$SETUP/lists/$s.meetings.txt"; done
python - "$DATA" "$NUM_TRAIN" <<'EOF'
import random, sys
d, n = sys.argv[1], int(sys.argv[2])
train = open(f"{d}/train.meetings.txt").read().split()
random.seed(0)                                   # same subset as prepare_data.py
open(f"{d}/download.txt", "w").write("\n".join(sorted(random.sample(train, n))) + "\n")
EOF
cat "$DATA/dev.meetings.txt" "$DATA/test.meetings.txt" >> "$DATA/download.txt"

# 3. Mix-Headset audio (16 kHz mono)
while read -r m; do
  [ -z "$m" ] && continue
  f="$DATA/wav/$m.Mix-Headset.wav"
  [ -s "$f" ] || wget -q -O "$f" "$AMI/AMICorpusMirror/amicorpus/$m/audio/$m.Mix-Headset.wav"
done < "$DATA/download.txt"

# 4. Reference RTTM / UEM for DER (only_words version)
while read -r m; do
  wget -q -O "$DATA/ref_rttm/$m.rttm" "$SETUP/only_words/rttms/test/$m.rttm"
  wget -q -O "$DATA/ref_uem/$m.uem"  "$SETUP/uems/test/$m.uem"
done < "$DATA/test.meetings.txt"
echo "done: $(ls "$DATA/wav" | wc -l) wav files"
