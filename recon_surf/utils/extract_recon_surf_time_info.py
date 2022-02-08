#!/usr/bin/env python3
import datetime
import dateutil.parser
import argparse
import yaml


def get_recon_all_stage_duration(line, previous_datetime_str):
    """
    Extract the duration of a recon-all stage from its log string.

    :param str line: line in recon-surf.log containing recon-all stage info.
        This must be of the form:
        #@# STAGE_NAME Fri Nov 26 15:51:40 UTC 2021
    :param str previous_datetime_str: datetime string of the previous recon-all stage

    :return: str stage_duration: stage duration in seconds
    """

    current_datetime_str = ' '.join(line.split()[-6:])
    current_date_time = dateutil.parser.parse(current_datetime_str)
    previous_date_time = dateutil.parser.parse(previous_datetime_str)
    stage_duration = (current_date_time - previous_date_time).total_seconds()

    return stage_duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_path', type=str,
                        default='scripts/recon-surf.log', 
                        help='Path to recon-surf.log file')
    parser.add_argument('-o', '--output_file_path', type=str,
                        default='', help='Path to output recon-surf_time.log file')
    parser.add_argument('--time_units', type=str,
                        default='m', help='Units of time [s, m]')
    args = parser.parse_args()

    lines = []
    with open(args.input_file_path) as file:
        for line in file:
            lines.append(line.rstrip())

    timestamp_feature = '@#@FSTIME'
    recon_all_stage_feature = '#@# '
    cmd_line_filter_phrases = ['done', 'Done', 'successful', 'finished without error', 'cmdline' ,'Running command', 'failed']
    filtered_cmds = ['ln ', 'rm ']

    if args.output_file_path == '':
        output_file_path = args.input_file_path.rsplit('/', 1)[0] + '/' + 'recon-surf_times.yaml'
    else:
        output_file_path = args.output_file_path 

    print('[INFO] Parsing file for recon_surf time information: {}\n'.format(args.input_file_path))
    if args.time_units not in ['s', 'm']:
        print('[WARN] Invalid time_units! Must be in s or m. Defaulting to m...')
        time_units = 'm'
    else:
        time_units = args.time_units

    yaml_dict = {}
    yaml_dict['date'] = lines[1]
    recon_surf_commands = []

    for i, line in enumerate(lines):
        ## Use recon_surf "stage" names as top level of recon-surf_commands entries:
        if '======' in line:
            stage_line = line
            current_recon_surf_stage_name = stage_line.strip('=')[1:-1].replace(' ', '-')
            if current_recon_surf_stage_name == 'DONE':
                continue
            recon_surf_commands.append({current_recon_surf_stage_name: []})

        if 'recon-surf.sh' in line and '--sid' in line:
            try:
                yaml_dict['subject_id'] = line.split()[line.split().index('--sid') + 1]
            except ValueError:
                print('[WARN] Could not extract subject ID from log file! It will not be added to the output.')

        ## Process lines containing the timestamp_feature:
        if timestamp_feature in line and 'cmdf' not in line:

            ## Parse out cmd name, start time, and duration:
            entry_dict = {}

            cmd_name = line.split()[2] + ' '
            if cmd_name in filtered_cmds:
                continue
            date_time_str = line.split()[1]
            start_time = date_time_str[11:]

            start_date_time = datetime.datetime.strptime(date_time_str, '%Y:%m:%d:%H:%M:%S')
            assert line.split()[5] == 'e'
            cmd_duration = float(line.split()[6])

            end_date_time = (start_date_time + datetime.timedelta(0, float(cmd_duration)))
            end_date_time_str = end_date_time.strftime('%Y:%m:%d:%H:%M:%S')
            end_time = end_date_time_str[11:]

            ## Get the line containing the actual full command:
            cmd_line_index = None
            cmd_line = None
            for previous_line_index in range(i-1, -1, -1):
                temp_line = lines[previous_line_index]
                if cmd_name in temp_line and all(phrase not in temp_line for phrase in cmd_line_filter_phrases):
                    cmd_line = temp_line
                    cmd_line_index = previous_line_index
                    break
            else:
                print('[WARN] Could not find the line containing the full command for {} in line {}! Skipping...\n'.format(cmd_name[:-1], i))
                continue

            entry_dict['cmd'] = cmd_line
            entry_dict['start'] = start_time
            entry_dict['stop'] = end_time
            if time_units == 's':
                entry_dict['duration_s'] = cmd_duration
            elif time_units == 'm':
                entry_dict['duration_m'] = round(cmd_duration / 60., 2)

            ## Parse out the same details for each stage in recon-all
            if cmd_name == 'recon-all ':
                entry_dict['stages'] = []
                first_stage = True

                for j in range(cmd_line_index, i):
                    if recon_all_stage_feature in lines[j] and len(lines[j].split()) > 5:
                        ## the second condition avoids lines such as "#@# 241395 lh 227149"

                        if not first_stage:
                            current_stage_start_time = lines[j].split()[-3]
                            stage_duration = get_recon_all_stage_duration(lines[j], previous_datetime_str)

                            stage_dict = {}
                            stage_dict['stage_name'] = stage_name
                            stage_dict['start'] = previous_stage_start_time
                            stage_dict['stop'] = current_stage_start_time
                            if time_units == 's':
                                stage_dict['duration_s'] = stage_duration
                            elif time_units == 'm':
                                stage_dict['duration_m'] = round(stage_duration / 60., 2)

                            entry_dict['stages'].append(stage_dict)
                        else:
                            first_stage = False

                        stage_name = ' '.join(lines[j].split()[:-6][1:])
                        previous_stage_start_time = lines[j].split()[-3]
                        previous_datetime_str = ' '.join(lines[j].split()[-6:])

                    ## Lines containing 'Ended' are used to find the end time of the last stage:
                    if 'Ended' in lines[j]:
                        current_stage_start_time = lines[j].split()[-3]
                        stage_duration = get_recon_all_stage_duration(lines[j], previous_datetime_str)

                        stage_dict = {}
                        stage_dict['stage_name'] = stage_name                        
                        stage_dict['start'] = previous_stage_start_time                        
                        stage_dict['stop'] = current_stage_start_time                        
                        if time_units == 's':
                            stage_dict['duration_s'] = stage_duration
                        elif time_units == 'm':
                            stage_dict['duration_m'] = round(stage_duration / 60., 2)
                            
                        entry_dict['stages'].append(stage_dict)

            recon_surf_commands[-1][current_recon_surf_stage_name].append(entry_dict)

    yaml_dict['recon-surf_commands'] = recon_surf_commands

    print('[INFO] Writing output to file: {}'.format(output_file_path))
    with open(output_file_path, 'w') as outfile:
        yaml.dump(yaml_dict, outfile, sort_keys=False)
