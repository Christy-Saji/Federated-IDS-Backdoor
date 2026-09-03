#!/bin/sh
# Fetch the 8 original CIC-IDS2017 MachineLearningCSV files into data/raw/.
#
#   sh scripts/preprocessing/fetch_cicids2017.sh
#
# UNB's own link (cicresearch.ca / 205.174.165.80) now 302s every scripted
# request to a registration form, so this pulls the same 8 files from the
# c01dsnap/CIC-IDS2017 mirror. That mirror was transcoded to UTF-8, so its Web
# Attack labels carry U+FFFD where the originals carry a Windows-1252 en-dash;
# flids/data/labels.py normalises both. Sizes are asserted below.
#
# Resumable: re-running continues a partial download rather than restarting.
set -e
BASE=https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DEST="$ROOT/data/raw"
mkdir -p "$DEST"
cd "$DEST"

# expected_size  filename
cat > .manifest <<'MANIFEST'
77123859 Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
76906168 Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
58316725 Friday-WorkingHours-Morning.pcap_ISCX.csv
176927918 Monday-WorkingHours.pcap_ISCX.csv
83102436 Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
52023263 Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
135078995 Tuesday-WorkingHours.pcap_ISCX.csv
225166395 Wednesday-workingHours.pcap_ISCX.csv
MANIFEST

while read -r size name; do
  echo ">>> $name"
  curl -fL -C - --retry 3 --retry-delay 2 -o "$name" "$BASE/$name"
  actual=$(wc -c < "$name" | tr -d ' ')
  if [ "$actual" != "$size" ]; then
    echo "SIZE MISMATCH for $name: got $actual, expected $size" >&2
    exit 1
  fi
done < .manifest
rm -f .manifest

echo
echo "All 8 files present and size-verified in $DEST"
echo "Next: python -m scripts.preprocessing.preprocess --data data/raw"
