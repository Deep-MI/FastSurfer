#!/usr/bin/env python3
import datetime
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file_path', type=str,
                        default='scripts/recon-surf.log', 
                        help='Path to recon-surf.log file')
    parser.add_argument('--out_file_path', type=str,
                        default='', help='Path to output recon-surf_time.log file')
    args = parser.parse_args()

    lines = []
    with open(args.in_file_path) as file:
        for line in file:
            lines.append(line.rstrip())

    timestamp_feature = '@#@FSTIME'
    recon_all_stage_feature = '#@# '
    cmd_line_filter_phrases = ['done', 'Done', 'successful', 'finished without error', 'cmdline' ,'Running command']

    if args.out_file_path == '':
        out_file_path = args.in_file_path.rsplit('/', 1)[0] + '/' + 'recon-surf_times.log'
    else:
        out_file_path = args.out_file_path 

    print('[INFO] Parsing file for recon_surf time information: {}'.format(args.in_file_path))

    output_file = open(out_file_path, 'w')
    output_file.write('Time log file for recon-surf.sh\n')

    for i, line in enumerate(lines):
        if i == 1:
            ## recon-surf datetime string:
            output_file.write(line + '\n')
        if '======' in line:
            ## recon-surf stage string:
            output_file.write('\n'+line+'\n')
        if timestamp_feature in line and 'cmdf' not in line:

            ## Parse out cmd name, start time, and duration:
            cmd_name = line.split(' ')[3] + ' '
            date_time_str = line.split(' ')[2]
            start_time = date_time_str[11:]

            start_date_time = datetime.datetime.strptime(date_time_str, '%Y:%m:%d:%H:%M:%S')
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

            ## Parse out the same details for each stage in recon-all
            if cmd_name == 'recon-all ':
                recon_all_stages = []
                first_stage = True

                for j in range(previous_line_index, i):
                    if recon_all_stage_feature in lines[j] and len(lines[j].split(' ')) > 5:
                        ## the second condition avoids the line "#@# 241395 lh 227149"

                        if not first_stage:
                            current_stage_start_time = lines[j].split(' ')[-3]
                            current_date_time = datetime.datetime.strptime(current_stage_start_time, '%H:%M:%S')
                            previous_date_time = datetime.datetime.strptime(previous_stage_start_time, '%H:%M:%S')
                            stage_time = (current_date_time - previous_date_time).total_seconds()

                            recon_all_stages.append({'stage_name':stage_name, 'start':previous_stage_start_time, 'end':current_stage_start_time, 'time':stage_time})
                        else:
                            first_stage = False

                        stage_name = ' '.join(lines[j].split(' ')[:-6][1:])
                        previous_stage_start_time = lines[j].split(' ')[-3]

                    ## Lines containing 'Ended' are used to find the end time of the last stage:
                    if 'Ended' in lines[j]:
                        current_stage_start_time = lines[j].split(' ')[-3]
                        current_date_time = datetime.datetime.strptime(current_stage_start_time, '%H:%M:%S')
                        previous_date_time = datetime.datetime.strptime(previous_stage_start_time, '%H:%M:%S')
                        stage_time = (current_date_time - previous_date_time).total_seconds()

                        recon_all_stages.append({'stage_name':stage_name, 'start':previous_stage_start_time, 'end':current_stage_start_time, 'time':stage_time})

            string = 'Command: {}\nStart: {}\nEnd: {}\nTime (m): {:.2f}'.format(temp_line, start_time,
                                                                               end_time, round(cmd_time / 60., 2))
            if cmd_name == 'recon-all ':
                string += '\n\nrecon-all stages:\n>>>'
                for recon_all_stage in recon_all_stages:
                    string += '\nStage: {}\nStart: {}\nEnd: {}\nTime (m): {:.2f}\n'.format(recon_all_stage['stage_name'], recon_all_stage['start'],
                                                                                         recon_all_stage['end'], round(recon_all_stage['time'] / 60., 2))
                string += '<<<'

            output_file.write('\n'+string+'\n')

    print('[INFO] Writing output to file: {}'.format(out_file_path))
    output_file.close()
