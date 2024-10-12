
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
    run_it_cmdf "$LF" "$CMDF" $cmd
  else
    run_it "$LF" $cmd
    if [ "${PIPESTATUS[0]}" -ne 0 ] ; then exit 1 ; fi
  fi
}

function run_it()
{
  # parameters
  # $1 : LF   (log file)
  # $@ : cmds (command to run)
  local LF=$1
  shift
  echo_quoted "$@" | tee -a "$LF"
  $timecmd "$@" 2>&1 | tee -a "$LF"
  if [ "${PIPESTATUS[0]}" -ne 0 ] ; then exit 1 ; fi
}

function run_it_cmdf()
{
  # parameters
  # $1 : LF   (log file)
  # $2 : CMDF (command file)
  # $@ : cmds (command to run)
  local LF=$1
  local CMDF=$2
  shift
  shift
  cmd="$(echo_quoted "$@" | tee -a "$LF")"
  printf -v tmp %q "$cmd"
  echo "echo $tmp" | tee -a "$CMDF"
  echo "$timecmd $cmd" | tee -a "$CMDF"
  echo "if [ \${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi" >> "$CMDF"
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

function check_allow_root()
{
  # Will check, if --allow_root is in arguments (to this function or the parent script) and
  # print an error message as well as exit.
  # Examples:
  # check_allow_root --arg 0 -> message and exit
  # some_script_which_calls_check_allow_root_without_parameters --flag -> message and exit

  local allow_root="false"
  for arg in "${BASH_ARGV[@]}" "$@" ; do if [[ "$arg" == "--allow_root" ]] ; then allow_root="true"; break ; fi ; done

  if [[ "$allow_root" != "true" ]] && [[ "$(id -u)" == "0" ]]
  then
    echo "ERROR: You are trying to run '$BASH_ARGV0' as root. We advice to avoid running FastSurfer"
    echo "  as root, because it will lead to files and folders created as root."
    echo "  If you are running FastSurfer in a docker container, you can specify the user"
    echo "  with '-u \$(id -u):\$(id -g)' (see https://docs.docker.com/engine/reference/run/#user)."
    echo "  If you want to force running as root, you may pass --allow_root to run_fastsurfer.sh."
    exit 1
  fi
}

function softlink_or_copy()
{
  # params
  # 1: file
  # 2: target
  # 3: logfile
  # 4: cmdf
  local LF="$3"
  local ln_cmd=(ln -sf "$1" "$2")
  local cp_cmd=(cp "$1" "$2")
  if [[ $# -eq 4 ]]
  then
    local CMDF=$4
    {
      echo "echo $(echo_quoted "${ln_cmd[@]}")"
      echo "$timecmd $(echo_quoted "${ln_cmd[@]}")"
      echo "if [ \${PIPESTATUS[0]} -ne 0 ]"
      echo "then"
      echo "  echo $(echo_quoted "${cp_cmd[@]}")"
      echo "  $timecmd $(echo_quoted "${cp_cmd[@]}")"
      echo "  if [ \${PIPESTATUS[0]} -ne 0 ] ; then exit 1 ; fi"
      echo "fi"
    } | tee -a "$CMDF"
  else
    {
      echo_quoted "${ln_cmd[@]}"
      $timecmd "${ln_cmd[@]}" 2>&1
      if [ "${PIPESTATUS[0]}" -ne 0 ]
      then
        echo_quoted "${cp_cmd[@]}"
        $timecmd "${cp_cmd[@]}" 2>&1
        if [ "${PIPESTATUS[0]}" -ne 0 ] ; then exit 1 ; fi
      fi
    } | tee -a "$LF"
  fi
}

function echo_quoted()
{
  # params ... 1-N
  sep=""
  for i in "$@"
  do
    if [[ "${i/ /}" != "$i" ]] ; then j="%q" ; else j="%s" ; fi
    printf "%s$j" "$sep" "$i"
    sep=" "
  done
  echo ""
}

function add_file_suffix()
{
  # params:
  # 1: filename
  # 2: suffix to add

  # example: add_file_suffix /path/to/file.nii.gz suffix -> /path/to/file.suffix.nii.gz

  # file extensions supported:
  file_extensions=(nii.gz nii mgz stats annot ctab label log txt lta xfm yaml)
  for extension in "${file_extensions[@]}"
  do
    pattern="\.${extension//./\\.}$"
    if [[ "$1" =~ $pattern ]]
    then
      length=$((${#1} - ${#extension}))
      echo "${1:0:$length}$2.$extension"
    fi
  done
}
