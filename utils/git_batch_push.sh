#!/bin/bash

# Broj datoteka po grupi
BATCH_SIZE=200
COUNTER=0

# Pronađi sve untracked datoteke
FILES=$(git ls-files --others --exclude-standard)

# Pretvori u array
FILES_ARRAY=($FILES)

# Ukupan broj datoteka
TOTAL=${#FILES_ARRAY[@]}

echo "Total untracked files: $TOTAL"

for ((i=0; i<TOTAL; i+=BATCH_SIZE)); do
    echo "Adding files $((i+1)) to $((i+BATCH_SIZE))..."

    # Dodaj batch
    git add "${FILES_ARRAY[@]:i:BATCH_SIZE}"

    # Commit s porukom
    git commit -m "Batch commit $((COUNTER+1))"

    # Pushaj
    git push origin main

    COUNTER=$((COUNTER+1))
done

echo "All done!"
