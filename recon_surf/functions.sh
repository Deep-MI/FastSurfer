
# set the binpath variable
if [ -z "$FASTSURFER_HOME" ]
then
  binpath="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/"
else
  binpath="$FASTSURFER_HOME/recon_surf/"
fi
export binpath

# fs_time command from fs60, fs72 fails in parallel mode, use local one
# also check for failure (e.g. on mac it fails)
timecmd="${binpath}fs_time"
$timecmd echo testing &> /dev/null
if [ "${PIPESTATUS[0]}" -ne 0 ] ; then
  echo "time command failing, not using time..."
  timecmd=""
fi
export timecmd

function RunIt()
{
  # parameters
  # $1 : cmd  (command to run)
  # $2 : LF   (log file)
  # $3 : CMDF (command file) optional
  # if CMDF is passed, then LF is ignored and cmd is echoed into CMDF and not run
  local cmd=$1
  local LF=$2
  if [[ $# -eq 3 ]]
  then
    local CMDF=$3
    printf -v tmp %q "$cmd"
    echo "echo $tmp" | tee -a $CMDF
    echo "$timecmd $cmd" | tee -a $CMDF
    echo "if [ \${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi" >> $CMDF
  else
    echo $cmd | tee -a $LF
    $timecmd $cmd 2>&1 | tee -a $LF
    if [ ${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi
  fi
}

function RunBatchJobs()
{
# parameters
# $1 : LF
# $2 ... : CMDFS
  local LOG_FILE=$1
  # launch jobs found in command files (shift past first logfile arg).
  # job output goes to a logfile named after the command file, which
  # later gets appended to LOG_FILE

  echo
  echo "RunBatchJobs: Logfile: $LOG_FILE"

  local PIDS=()
  local LOGS=()
  shift
  local JOB
  local LOG
  for cmdf in "$@"; do
    echo "RunBatchJobs: CMDF: $cmdf"
    chmod u+x "$cmdf"
    JOB="$cmdf"
    LOG=$cmdf.log
    echo "" >& "$LOG"
    echo " $JOB" >> "$LOG"
    echo "" >> "$LOG"
    exec "$JOB" >> "$LOG" 2>&1 &
    PIDS=("${PIDS[@]}" "$!")
    LOGS=("${LOGS[@]}" "$LOG")

  done
  # wait till all processes have finished
  local PIDS_STATUS=()
  for pid in "${PIDS[@]}"; do
    echo "Waiting for PID $pid of (${PIDS[*]}) to complete..."
    wait "$pid"
    PIDS_STATUS=("${PIDS_STATUS[@]}" "$?")
  done
  # now append their logs to the main log file
  for log in "${LOGS[@]}"
  do
    cat "$log" >> "$LOG_FILE"
    rm -f "$log"
  done
  echo "PIDs (${PIDS[*]}) completed and logs appended."
  # and check for failures
  for pid_status in "${PIDS_STATUS[@]}"
  do
    if [ "$pid_status" != "0" ] ; then
      exit 1
    fi
  done
}

function softlink_or_copy()
{
  # params
  # 1: file
  # 2: target
  # 3: logfile
  # 4: cmdf
  local LF="$3"
  local ln_cmd="ln -sf $1 $2"
  local cp_cmd="cp $1 $2"
  if [[ $# -eq 4 ]]
  then
    local CMDF=$4
    echo "echo \"$ln_cmd\" " | tee -a $CMDF
    echo "$timecmd $ln_cmd " | tee -a $CMDF
    echo "if [ \${PIPESTATUS[0]} -ne 0 ]" | tee -a $CMDF
    echo "then " | tee -a $CMDF
    echo "  echo \"$cp_cmd\" " | tee -a $CMDF
    echo "  $timecmd $cp_cmd " | tee -a $CMDF
    echo "  if [ \${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi" >> $CMDF
    echo "fi" | tee -a $CMDF
  else
    echo $ln_cmd | tee -a $LF
    $timecmd $ln_cmd 2>&1 | tee -a $LF
    if [ ${PIPESTATUS[0]} -ne 0 ]
    then
      echo $cp_cmd | tee -a $LF
      $timecmd $cp_cmd 2>&1 | tee -a $LF
      if [ ${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi
    fi
  fi
}
