#!/usr/bin/env python3
import datetime
import argparse
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file_path', type=str,
                        default='scripts/recon-surf.log', 
                        help='Path to recon-surf.log file')
    parser.add_argument('--out_file_path', type=str,
                        default='', help='Path to output recon-surf_time.log file')
    parser.add_argument('--time_units', type=str,
                        default='m', help='Units of time [s, m]')
    args = parser.parse_args()

    lines = []
    with open(args.in_file_path) as file:
        for line in file:
            lines.append(line.rstrip())

    timestamp_feature = '@#@FSTIME'
    recon_all_stage_feature = '#@# '
    cmd_line_filter_phrases = ['done', 'Done', 'successful', 'finished without error', 'cmdline' ,'Running command']
    filtered_cmds = ['ln ', 'rm ']

    if args.out_file_path == '':
        out_file_path = args.in_file_path.rsplit('/', 1)[0] + '/' + 'recon-surf_times.yaml'
    else:
        out_file_path = args.out_file_path 

    print('[INFO] Parsing file for recon_surf time information: {}'.format(args.in_file_path))
    if args.time_units not in ['s', 'm']:
        print('[WARN] Invalid time_units! Must be in s or m. Defaulting to m...')
        time_units = 'm'
    else:
        time_units = args.time_units

    yaml_dict = {}
    yaml_dict['date'] = lines[1]
    yaml_dict['recon-surf_commands'] = []

    for i, line in enumerate(lines):
        ## Use recon_surf "stage" names as top level of recon-surf_commands entries:
        if '======' in line:
            stage_line = line
            current_recon_surf_stage_name = stage_line.strip('=')[1:-1].replace(' ', '-')
            if current_recon_surf_stage_name == 'DONE':
                continue
            yaml_dict['recon-surf_commands'].append({current_recon_surf_stage_name: []})

        ## Process lines containing the timestamp_feature:
        if timestamp_feature in line and 'cmdf' not in line:

            ## Parse out cmd name, start time, and duration:
            entry_dict = {}

            cmd_name = line.split(' ')[3] + ' '
            if cmd_name in filtered_cmds:
                continue
            date_time_str = line.split(' ')[2]
            start_time = date_time_str[11:]

            start_date_time = datetime.datetime.strptime(date_time_str, '%Y:%m:%d:%H:%M:%S')
            assert line.split(' ')[6] == 'e'
            cmd_time = float(line.split(' ')[7])

            end_date_time = (start_date_time + datetime.timedelta(0, float(cmd_time)))
            end_date_time_str = end_date_time.strftime('%Y:%m:%d:%H:%M:%S')
            end_time = end_date_time_str[11:]

            ## Get the line containing the actual full command:
            previous_line_index = i
            cmd_line = None
            while True:
                temp_line = lines[previous_line_index - 1]

                if cmd_name in temp_line and all(phrase not in temp_line for phrase in cmd_line_filter_phrases):
                    break
                else:
                    previous_line_index -= 1

            cmd_line = temp_line

            entry_dict['cmd'] = cmd_line
            entry_dict['start'] = start_time
            entry_dict['stop'] = end_time
            if time_units == 's':
                entry_dict['duration_s'] = cmd_time
            elif time_units == 'm':
                entry_dict['duration_m'] = round(cmd_time / 60., 2)

            ## Parse out the same details for each stage in recon-all
            if cmd_name == 'recon-all ':
                entry_dict['stages'] = []
                first_stage = True

                for j in range(previous_line_index, i):
                    if recon_all_stage_feature in lines[j] and len(lines[j].split(' ')) > 5:
                        ## the second condition avoids lines such as "#@# 241395 lh 227149"

                        if not first_stage:
                            current_stage_start_time = lines[j].split(' ')[-3]

                            ## Reconstruct string for easier parsing of stage date and time:
                            ## (Example: "Nov 14 2021 12:31:34")
                            current_datetime_str = lines[j].split(' ')[-5] + ' ' + \
                                                   lines[j].split(' ')[-4] + ' ' + \
                                                   lines[j].split(' ')[-1] + ' ' + \
                                                   lines[j].split(' ')[-3]
                            current_date_time = datetime.datetime.strptime(current_datetime_str, '%b %d %Y %H:%M:%S')
                            previous_date_time = datetime.datetime.strptime(previous_datetime_str, '%b %d %Y %H:%M:%S')
                            stage_time = (current_date_time - previous_date_time).total_seconds()

                            stage_dict = {}
                            stage_dict['stage_name'] = stage_name
                            stage_dict['start'] = previous_stage_start_time
                            stage_dict['stop'] = current_stage_start_time
                            if time_units == 's':
                                stage_dict['duration_s'] = stage_time
                            elif time_units == 'm':
                                stage_dict['duration_m'] = round(stage_time / 60., 2)

                            entry_dict['stages'].append(stage_dict)
                        else:
                            first_stage = False

                        stage_name = ' '.join(lines[j].split(' ')[:-6][1:])
                        previous_stage_start_time = lines[j].split(' ')[-3]
                        previous_datetime_str = lines[j].split(' ')[-5] + ' ' + \
                                                lines[j].split(' ')[-4] + ' ' + \
                                                lines[j].split(' ')[-1] + ' ' + \
                                                lines[j].split(' ')[-3]

                    ## Lines containing 'Ended' are used to find the end time of the last stage:
                    if 'Ended' in lines[j]:
                        current_stage_start_time = lines[j].split(' ')[-3]

                        current_datetime_str = lines[j].split(' ')[-5] + ' ' + \
                                               lines[j].split(' ')[-4] + ' ' + \
                                               lines[j].split(' ')[-1] + ' ' + \
                                               lines[j].split(' ')[-3]
                        current_date_time = datetime.datetime.strptime(current_datetime_str, '%b %d %Y %H:%M:%S')
                        previous_date_time = datetime.datetime.strptime(previous_datetime_str, '%b %d %Y %H:%M:%S')
                        stage_time = (current_date_time - previous_date_time).total_seconds()

                        stage_dict = {}
                        stage_dict['stage_name'] = stage_name                        
                        stage_dict['start'] = previous_stage_start_time                        
                        stage_dict['stop'] = current_stage_start_time                        
                        if time_units == 's':
                            stage_dict['duration_s'] = stage_time
                        elif time_units == 'm':
                            stage_dict['duration_m'] = round(stage_time / 60., 2)
                            
                        entry_dict['stages'].append(stage_dict)

            yaml_dict['recon-surf_commands'][-1][current_recon_surf_stage_name].append(entry_dict)

    print('[INFO] Writing output to file: {}'.format(out_file_path))
    with open(out_file_path, 'w') as outfile:
        yaml.dump(yaml_dict, outfile, sort_keys=False)
