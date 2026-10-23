#!/usr/bin/env bash
# Stitch the recorded clips into mogscan_demo.mp4 with a generated title card.
# Usage: put clips named as in script.md into video/clips/, then run ./assemble.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
# title.png is pre-rendered (this ffmpeg build has no drawtext filter).
ffmpeg -y -loglevel error -loop 1 -framerate 30 -t 4 -i title.png -f lavfi -t 4 -i anullsrc=r=48000:cl=stereo \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest build/00_title.mp4

: > build/list.txt
add() { echo "file '$1'" >> build/list.txt; }
first=1
for clip in clips/*.mov clips/*.mp4; do
  [ -e "$clip" ] || continue
  base=$(basename "${clip%.*}")
  vf="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x07070b,fps=30"
  if ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$clip" | grep -q .; then
    ffmpeg -y -loglevel error -i "$clip" -vf "$vf" -map 0:v:0 -map 0:a:0 -c:v libx264 -pix_fmt yuv420p \
      -c:a aac -ar 48000 -ac 2 "build/$base.mp4"
  else  # screen recordings often have no audio track; add silence so concat works
    ffmpeg -y -loglevel error -i "$clip" -f lavfi -i anullsrc=r=48000:cl=stereo -vf "$vf" -map 0:v:0 -map 1:a \
      -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "build/$base.mp4"
  fi
  add "$base.mp4"
  # Title card goes after the cold-open hook.
  if [ $first = 1 ]; then add "00_title.mp4"; first=0; fi
done
ffmpeg -y -f concat -safe 0 -i build/list.txt -c copy mogscan_demo.mp4
echo "wrote video/mogscan_demo.mp4"
