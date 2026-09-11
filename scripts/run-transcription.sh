#!/usr/bin/env bash
# Send an audio file to a running WhisperX API server and print the result.
#
# Usage:
#   server/try-transcribe.sh AUDIO_FILE [options]
#
# Options:
#   -u, --url URL          Base URL of the server        (default: http://localhost:8000)
#   -f, --format FORMAT     json | verbose_json | text | srt | vtt   (default: verbose_json)
#   -l, --language CODE     ISO language code (omit to use the server default / auto)
#   -p, --prompt TEXT       initial_prompt / vocabulary hint
#       --translate         hit /v1/audio/translations instead (X -> English)
#   -d, --diarize           request speaker diarization (server needs HF_TOKEN)
#       --min-speakers N
#       --max-speakers N
#       --words             ask for word-level timestamps (verbose_json only)
#   -k, --api-key KEY       bearer token (default: $API_KEY env var)
#   -h, --help
#
# Examples:
#   server/try-transcribe.sh sample.wav
#   server/try-transcribe.sh meeting.m4a -d --min-speakers 2 --max-speakers 4 --words
#   API_KEY=secret server/try-transcribe.sh talk.mp3 -f srt -u http://whisperx.lan:8000

set -euo pipefail

URL="http://localhost:8989"
FORMAT="srt"
LANGUAGE="en"
PROMPT=""
ENDPOINT="/v1/audio/transcriptions"
DIARIZE="1"
MIN_SPEAKERS="1"
MAX_SPEAKERS=""
WORDS=""
API_KEY=""
AUDIO=""

die() { echo "error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    -u|--url)          URL="$2"; shift 2 ;;
    -f|--format)       FORMAT="$2"; shift 2 ;;
    -l|--language)     LANGUAGE="$2"; shift 2 ;;
    -p|--prompt)       PROMPT="$2"; shift 2 ;;
    --translate)       ENDPOINT="/v1/audio/translations"; shift ;;
    -d|--diarize)      DIARIZE="1"; shift ;;
    --min-speakers)    MIN_SPEAKERS="$2"; shift 2 ;;
    --max-speakers)    MAX_SPEAKERS="$2"; shift 2 ;;
    --words)           WORDS="1"; shift ;;
    -k|--api-key)      API_KEY="$2"; shift 2 ;;
    -h|--help)         sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)                die "unknown option: $1" ;;
    *)                 [ -z "$AUDIO" ] || die "unexpected extra argument: $1"; AUDIO="$1"; shift ;;
  esac
done

[ -n "$AUDIO" ] || die "no audio file given (see --help)"
[ -f "$AUDIO" ] || die "no such file: $AUDIO"
command -v curl >/dev/null || die "curl is required"

form=(-F "file=@${AUDIO}" -F "response_format=${FORMAT}" -F "model=whisper-1")
[ -n "$LANGUAGE" ]     && form+=(-F "language=${LANGUAGE}")
[ -n "$PROMPT" ]       && form+=(-F "prompt=${PROMPT}")
[ -n "$DIARIZE" ]      && form+=(-F "diarize=true")
[ -n "$MIN_SPEAKERS" ] && form+=(-F "min_speakers=${MIN_SPEAKERS}")
[ -n "$MAX_SPEAKERS" ] && form+=(-F "max_speakers=${MAX_SPEAKERS}")
[ -n "$WORDS" ]        && form+=(-F "timestamp_granularities[]=word")

auth=()
[ -n "$API_KEY" ] && auth=(-H "Authorization: Bearer ${API_KEY}")

echo ">> POST ${URL}${ENDPOINT}  (file=$(basename "$AUDIO"), format=${FORMAT}${DIARIZE:+, diarize})" >&2

body_file="$(mktemp)"
trap 'rm -f "$body_file"' EXIT
start=$(date +%s)
http_code=$(
  curl -sS -X POST "${URL}${ENDPOINT}" \
    "${auth[@]}" "${form[@]}" \
    -o "$body_file" -w '%{http_code}' || true
)
elapsed=$(( $(date +%s) - start ))
[ -n "$http_code" ] || http_code="000"
echo ">> HTTP ${http_code} in ${elapsed}s" >&2

case "$FORMAT" in
  json|verbose_json)
    if command -v python3 >/dev/null; then
      python3 -m json.tool < "$body_file" || cat "$body_file"
    else
      cat "$body_file"
    fi
    ;;
  *)
    cat "$body_file"
    ;;
esac

[ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]
