#!/bin/bash

# Counter for numbering the samples
counter=1

# Loop through each file matching the pattern
for file in vlcsnap-*; do
    # Get the extension of the file
    extension="${file##*.}"
    # Rename the file to sample-counter.extension
    mv "$file" "sample-$counter.$extension"
    # Increment the counter
    ((counter++))
done

