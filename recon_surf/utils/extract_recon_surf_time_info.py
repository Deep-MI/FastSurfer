#!/usr/bin/env python3
from datetime import datetime, timedelta
from pathlib import Path

import dateutil.parser
import argparse
import yaml
import locale


def get_recon_all_stage_duration(line: str, previous_datetime_str: str) -> float:
    """Extract the duration of a recon-all stage from its log string.

    Parameters
    ----------
    line : str
        line in recon-surf.log containing recon-all stage info.
        This must be of the form:
        #@# STAGE_NAME Fri Nov 26 15:51:40 UTC 2021
    previous_datetime_str : str
        datetime string of the previous recon-all stage

    Returns
    -------
    stage_duration
        stage duration in seconds

    """

    current_datetime_str = " ".join(line.split()[-6:])
    try:
        current_date_time = dateutil.parser.parse(current_datetime_str)
        previous_date_time = dateutil.parser.parse(previous_datetime_str)
    except ParserError: # strptime considers the computers time locale settings
        locale.setlocale(locale.LC_TIME,"")
        current_date_time = datetime.strptime(current_datetime_str, "%a %d. %b %H:%M:%S %Z %Y")
        previous_date_time = datetime.strptime(previous_datetime_str, "%a %d. %b %H:%M:%S %Z %Y")
    stage_duration = (current_date_time - previous_date_time).total_seconds()

    return stage_duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file_path",
        type=Path,
        default="scripts/recon-surf.log",
        help="Path to recon-surf.log file",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        type=Path,
        default=None,
        help="Path to output recon-surf_time.log file",
    )
    parser.add_argument(
        "--time_units",
        choices=["m", "s"],
        default="m",
        help="Units of time [s, m]",
    )
    args = parser.parse_args()

    with open(args.input_file_path) as file:
        lines = [line.rstrip() for line in file.readlines()]

    timestamp_feature = "@#@FSTIME"
    recon_all_stage_feature = "#@# "
    cmd_line_filter_phrases = [
        "done",
        "Done",
        "successful",
        "finished without error",
        "cmdline",
        "Running command",
        "failed",
        "FSRUNTIME@",
        "ru_nivcsw",
        "ru_nvcsw",
        "ru_nsignals",
        "ru_msgrcv",
        "ru_msgsnd",
        "ru_oublock",
        "ru_inblock",
        "ru_nswap",
        "ru_majflt",
        "ru_minflt",
        "ru_isrss",
        "ru_idrss",
        "ru_ixrss",
        "ru_maxrss",
        "stimesec",
        "utimesec",
        "#",
        "This may cause",
    ]
    filtered_cmds = ["ln", "rm", "cp"]

    if not args.output_file_path:
        output_file_path = args.input_file_path.parent / "recon-surf_times.yaml"
    else:
        output_file_path = args.output_file_path

    print(
        f"[INFO] Parsing file for recon_surf time information: {args.input_file_path}\n"
    )
    if args.time_units not in ["s", "m"]:
        print("[WARN] Invalid time_units! Must be in s or m. Defaulting to m...")
        time_units = "m"
    else:
        time_units = args.time_units

    yaml_dict = {"date": lines[1]}
    pre_recon_surf_stage_name = "Starting up / no stage defined yet"
    current_recon_surf_stage_name = pre_recon_surf_stage_name
    recon_surf_commands = [{current_recon_surf_stage_name: []}]

    for i, line in enumerate(lines):
        ## Use recon_surf "stage" names as top level of recon-surf_commands entries:
        if "======" in line and "teration" not in line:
            stage_line = line
            current_recon_surf_stage_name = stage_line.strip("= ").replace(" ", "-")
            if current_recon_surf_stage_name == "DONE":
                continue
            recon_surf_commands.append({current_recon_surf_stage_name: []})

        line_parts = line.split()
        if "recon-surf.sh" in line and "--sid" in line:
            try:
                yaml_dict["subject_id"] = line_parts[line_parts.index("--sid") + 1]
            except ValueError:
                print(
                    "[WARN] Could not extract subject ID from log file! It will not be added to the output."
                )

        ## Process lines containing the timestamp_feature:
        if timestamp_feature in line and "cmdf" not in line:

            ## Parse out cmd name, start time, and duration:
            entry_dict = {}

            cmd_name = line_parts[2]
            if cmd_name in filtered_cmds:
                continue
            date_time_str = line_parts[1]
            start_time = date_time_str[11:]

            start_date_time = datetime.strptime(
                date_time_str, "%Y:%m:%d:%H:%M:%S"
            )
            assert line_parts[5] == "e"
            cmd_duration = float(line_parts[6])

            end_date_time = start_date_time + timedelta(0, float(cmd_duration))
            end_date_time_str = end_date_time.strftime("%Y:%m:%d:%H:%M:%S")
            end_time = end_date_time_str[11:]

            ## Get the line containing the actual full command:
            cmd_line_index = None
            cmd_line = None
            for previous_line_index in range(i - 1, -1, -1):
                temp_line = lines[previous_line_index]
                if cmd_name + " " in temp_line and all(
                    phrase not in temp_line for phrase in cmd_line_filter_phrases
                ):
                    cmd_line = temp_line
                    cmd_line_index = previous_line_index
                    break
            else:
                print(
                    f"[WARN] Could not find the line containing the full command for "
                    f"{cmd_name} in line {i+1}! Skipping...\n"
                )
                continue

            entry_dict["cmd"] = cmd_line.lstrip()
            entry_dict["start"] = start_time
            entry_dict["stop"] = end_time
            if time_units == "s":
                entry_dict["duration_s"] = cmd_duration
            elif time_units == "m":
                entry_dict["duration_m"] = round(cmd_duration / 60.0, 2)

            ## Parse out the same details for each stage in recon-all
            if cmd_name == "recon-all":
                entry_dict["stages"] = []
                first_stage = True
                previous_datetime_str = ""
                stage_name = ""
                previous_stage_start_time = ""

                for j in range(cmd_line_index, i):
                    if (
                        recon_all_stage_feature in lines[j]
                        and len(lines[j].split()) > 5
                    ):
                        ## the second condition avoids lines such as "#@# 241395 lh 227149"

                        if not first_stage:
                            current_stage_start_time = lines[j].split()[-3]
                            stage_duration = get_recon_all_stage_duration(
                                lines[j], previous_datetime_str
                            )

                            stage_dict = {}
                            stage_dict["stage_name"] = stage_name
                            stage_dict["start"] = previous_stage_start_time
                            stage_dict["stop"] = current_stage_start_time
                            if time_units == "s":
                                stage_dict["duration_s"] = stage_duration
                            elif time_units == "m":
                                stage_dict["duration_m"] = round(
                                    stage_duration / 60.0, 2
                                )

                            entry_dict["stages"].append(stage_dict)
                        else:
                            first_stage = False

                        stage_name = " ".join(lines[j].split()[:-6][1:])
                        previous_stage_start_time = lines[j].split()[-3]
                        previous_datetime_str = " ".join(lines[j].split()[-6:])

                    ## Lines containing 'Ended' are used to find the end time of the last stage:
                    if "Ended" in lines[j]:
                        current_stage_start_time = lines[j].split()[-3]
                        stage_duration = get_recon_all_stage_duration(
                            lines[j], previous_datetime_str
                        )

                        stage_dict = {}
                        stage_dict["stage_name"] = stage_name
                        stage_dict["start"] = previous_stage_start_time
                        stage_dict["stop"] = current_stage_start_time
                        if time_units == "s":
                            stage_dict["duration_s"] = stage_duration
                        elif time_units == "m":
                            stage_dict["duration_m"] = round(stage_duration / 60.0, 2)

                        entry_dict["stages"].append(stage_dict)

            recon_surf_commands[-1][current_recon_surf_stage_name].append(entry_dict)

    yaml_dict["recon-surf_commands"] = recon_surf_commands

    print(f"[INFO] Writing output to file: {output_file_path}")
    with open(output_file_path, "w") as outfile:
        yaml.dump(yaml_dict, outfile, sort_keys=False)
