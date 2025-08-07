#!/bin/bash

# Usage: ./extract_avg_runtime.sh input.txt

if [ $# -ne 1 ]; then
  echo "Usage: $0 input_file.txt"
  exit 1
fi

input_file="$1"

awk '
  /runtime =/ {
    # Split by "runtime = " and take the second part
    split($0, parts, "runtime = ");
    split(parts[2], ms, "ms");
    sum += ms[1];
    count++;
  }
  END {
    if (count > 0) {
      printf("Average runtime = %.2f ms\n", sum / count);
    } else {
      print "No runtimes found.";
    }
  }
' "$input_file"

