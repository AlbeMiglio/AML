#!/bin/bash
#
# Build the submission archive: the sources tracked in git plus the compiled
# report, and nothing else.
#
# Archiving the working directory is not safe here: it also carries whatever
# untracked material happens to sit in the tree (old figures, previous PDFs,
# results folders, downloaded weights). We therefore export from git, which by
# construction contains exactly the files that belong to the project.

set -euo pipefail
cd "$(dirname "$0")"

OUTPUT_FILE="aml_submission.zip"
PDF="report/main.pdf"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: tracked files have uncommitted changes; the archive will contain"
    echo "         the committed version. Commit first if that is not what you want."
    echo
fi

rm -f "$OUTPUT_FILE"
git archive --format=zip --output="$OUTPUT_FILE" HEAD

if [ -f "$PDF" ]; then
    zip -q "$OUTPUT_FILE" "$PDF"
    echo "Added compiled report: $PDF"
else
    echo "WARNING: $PDF not found — compile the report before submitting."
fi

echo "--------------------------------------------------"
echo "Archive: $OUTPUT_FILE ($(du -h "$OUTPUT_FILE" | cut -f1))"
echo "Contents: $(unzip -l "$OUTPUT_FILE" | tail -1 | awk '{print $2}') files"
echo
echo "Dataset and weights are intentionally excluded; retrieve them with"
echo "download_data_and_weights.sh."
echo "--------------------------------------------------"
