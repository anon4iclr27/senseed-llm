# Sourced by the run scripts: mirrors all output to a log file and prints a
# "[k/N] stage" line with elapsed time and a rough ETA before each stage.
#   progress_init <name> <n_stages>     log goes to logs/<name>-<timestamp>.log
#   stage "<label>"                     call at the start of every stage
progress_init() {
  mkdir -p logs
  LOG_FILE="logs/$1-$(date +%Y%m%d-%H%M%S).log"
  N_STAGES=$2; K_STAGE=0; T_START=$(date +%s)
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "log: $LOG_FILE   (follow with: tail -f $LOG_FILE)"
}
_hms() { printf '%d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }
stage() {
  K_STAGE=$((K_STAGE+1)); local now=$(date +%s) el eta=""
  el=$((now-T_START))
  [ "$K_STAGE" -gt 1 ] && eta="  ETA $(_hms $((el*(N_STAGES-K_STAGE+1)/(K_STAGE-1))))"
  printf '\n[%d/%d] %s  (%s elapsed%s)\n' "$K_STAGE" "$N_STAGES" "$1" "$(_hms $el)" "$eta"
}
progress_done() { printf '\ndone: %s total; log %s\n' "$(_hms $(($(date +%s)-T_START)))" "$LOG_FILE"; }
