#!/bin/bash

# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Thread budget functions, sourced by functions.sh. Kept separate so that scripts can source
# them without the FreeSurfer setup and fs_time probe of functions.sh.
#
# One thread budget per stage, from the --threads flag, else from OMP_NUM_THREADS, else a default.
# Every library reads its own variable, so all of them are exported from that budget. A library
# variable the user set lower stays a ceiling for that library. OMP_NUM_THREADS is the budget
# itself and is replaced like the others.

thread_env_vars=(OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS
                 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS)

# the user's values, before this process exports its own; restore_thread_env puts them back
thread_env_user_set=""
for thread_var in "${thread_env_vars[@]}" OMP_THREAD_LIMIT ; do
  if printenv "$thread_var" > /dev/null ; then
    printf -v "thread_env_user_$thread_var" '%s' "$(printenv "$thread_var")"
    thread_env_user_set+=" $thread_var"
  fi
done

function positive_int()
{
  # USAGE: positive_int <value>
  # Prints the first entry of an OpenMP style list such as "4,2", if it is a positive integer.
  local first="${1%%,*}"
  if [[ "$first" =~ ^[1-9][0-9]*$ ]] ; then echo "$first" ; fi
}

function available_cpus()
{
  # The number of CPUs this process may use. Not plain nproc: GNU nproc returns OMP_NUM_THREADS
  # when that is set, and macOS has no nproc.
  local n=""
  if command -v nproc > /dev/null ; then n="$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2> /dev/null)" ; fi
  if [[ -z "$(positive_int "$n")" ]] ; then n="$(sysctl -n hw.ncpu 2> /dev/null)" ; fi
  if [[ -z "$(positive_int "$n")" ]] ; then n="$(getconf _NPROCESSORS_ONLN 2> /dev/null)" ; fi
  if [[ -z "$(positive_int "$n")" ]] ; then n=1 ; fi
  echo "$n"
}

function resolve_threads()
{
  # USAGE: resolve_threads <value passed to --threads, empty if none> <default>
  # Sets threads_budget, threads_source and threads_note. Returns 1 for an invalid flag value.
  # max, 0 and negative values mean all available CPUs.
  local flag user_omp limit
  flag="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  threads_note=""
  if [[ "$flag" =~ ^(max|-[0-9]+|0+)$ ]] ; then
    threads_budget="$(available_cpus)" ; threads_source="--threads $1"
  elif [[ -n "$flag" ]] ; then
    threads_budget="$(positive_int "$flag")" ; threads_source="--threads"
    if [[ "$threads_budget" != "$flag" ]] ; then
      threads_note="ERROR: Invalid number of threads '$1', must be a positive integer or 'max'."
      return 1
    fi
  else
    user_omp="$(positive_int "$thread_env_user_OMP_NUM_THREADS")"
    if [[ -n "$user_omp" ]] ; then
      threads_budget="$user_omp" ; threads_source="OMP_NUM_THREADS"
    else
      if [[ " $thread_env_user_set " == *" OMP_NUM_THREADS "* ]] ; then
        threads_note="WARNING: Ignoring OMP_NUM_THREADS='$thread_env_user_OMP_NUM_THREADS', which is not a positive number."
      fi
      threads_budget="$2" ; threads_source="default"
    fi
  fi
  limit="$(positive_int "$thread_env_user_OMP_THREAD_LIMIT")"
  if [[ -n "$limit" ]] && [[ "$limit" -lt "$threads_budget" ]] ; then
    threads_budget="$limit" ; threads_source+=", capped by OMP_THREAD_LIMIT"
  fi
}

function thread_env_value()
{
  # USAGE: thread_env_value <variable> <threads>
  # Prints the value to export: <threads>, or the user's value of a library variable if lower.
  local name="thread_env_user_$1" user
  user="$(positive_int "${!name}")"
  if [[ "$1" != "OMP_NUM_THREADS" ]] && [[ -n "$user" ]] && [[ "$user" -lt "$2" ]] ; then echo "$user"
  else echo "$2"
  fi
}

function thread_env_exports()
{
  # USAGE: thread_env_exports <threads>
  # Prints the export lines, for scripts written to a command file.
  local var
  for var in "${thread_env_vars[@]}" ; do echo "export $var=$(thread_env_value "$var" "$1")" ; done
}

function set_thread_env()
{
  # USAGE: set_thread_env <threads>
  # Exports the thread variables and sets threads_capped to those a user value held lower.
  local var value
  threads_capped=""
  for var in "${thread_env_vars[@]}" ; do
    value="$(thread_env_value "$var" "$1")"
    export "$var=$value"
    if [[ "$value" != "$1" ]] ; then threads_capped+=" $var=$value" ; fi
  done
}

function restore_thread_env()
{
  # Puts the thread variables back as the user's environment had them.
  local var name
  for var in "${thread_env_vars[@]}" ; do
    name="thread_env_user_$var"
    if [[ " $thread_env_user_set " == *" $var "* ]] ; then export "$var=${!name}"
    else unset "$var"
    fi
  done
}

function describe_threads()
{
  # USAGE: describe_threads <what>
  # The log line naming the budget, where it came from and what the environment capped.
  echo "INFO: $1 uses $threads_budget threads (from $threads_source)."
  if [[ -n "$threads_capped" ]] ; then echo "  Kept lower by the environment:$threads_capped" ; fi
  if [[ -n "$threads_note" ]] ; then echo "$threads_note" ; fi
}
