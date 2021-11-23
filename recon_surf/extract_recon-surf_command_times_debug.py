#!/usr/bin/env python3
import datetime
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file_path', type=str,
                        default='/home/abdelrahmana/workspace/testing_output/002-3/241395/scripts/recon-surf.log', 
                        help='Path to recon-surf.log file')
    parser.add_argument('--out_file_path', type=str,
                        default='', help='Path to output recon-surf_time.log file')
    parser.add_argument('--print_result', dest='print_result',
                        action='store_true', help='Whether to print the result')
    parser.set_defaults(print_result=False)
    args = parser.parse_args()

    lines = []
    with open(args.in_file_path) as file:
        for line in file:
            lines.append(line.rstrip())

    timestamp_feature = '@#@FSTIME'
    recon_all_stage_feature = '#@# '
    cmd_line_filter_phrases = ['done', 'Done', 'successful', 'finished without error', 'cmdline' ,'Running command']
    version = 1

    if args.out_file_path == '':
        out_file_path = args.in_file_path.rsplit('/', 1)[0] + '/' + 'recon-surf_times.log'
    else:
        out_file_path = args.out_file_path 

    debug = False

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
            if debug:
                print('Line number: {}'.format(i))
                print('Line:\n', line); print()
                print('Next line:\n', lines[i+1]); print()

            ## Parse out cmd name, start time, and duration:
            cmd_name = line.split(' ')[3] + ' '
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

            ## Parse out the same details for each stage in recon-all
            if cmd_name == 'recon-all ':
                recon_all_stages = []
                if debug:
                    print('\n>>recon-all stages:')
                first_stage = True

                for j in range(previous_line_index, i):
                    if recon_all_stage_feature in lines[j] and len(lines[j].split(' ')) > 5:
                        ## the second condition avoids the line "#@# 241395 lh 227149"

                        if not first_stage:
                            current_stage_start_time = lines[j].split(' ')[-3]
                            current_date_time = datetime.datetime.strptime(current_stage_start_time, '%H:%M:%S')
                            previous_date_time = datetime.datetime.strptime(previous_stage_start_time, '%H:%M:%S')
                            stage_time = (current_date_time - previous_date_time).total_seconds()

                            if debug:
                                print('Start: {}'.format(current_stage_start_time))
                                print('Time (s): {}'.format(stage_time))
                                print()

                            recon_all_stages.append({'stage_name':stage_name, 'start':previous_stage_start_time, 'end':current_stage_start_time, 'time':stage_time})
                        else:
                            first_stage = False

                        stage_name = ' '.join(lines[j].split(' ')[:-6][1:])
                        previous_stage_start_time = lines[j].split(' ')[-3]

                        if debug:
                            print('Stage name: {}'.format(stage_name))

                    ## Lines containing 'Ended' are used to find the end time of the last stage:
                    if 'Ended' in lines[j]:
                        current_stage_start_time = lines[j].split(' ')[-3]
                        current_date_time = datetime.datetime.strptime(current_stage_start_time, '%H:%M:%S')
                        previous_date_time = datetime.datetime.strptime(previous_stage_start_time, '%H:%M:%S')
                        stage_time = (current_date_time - previous_date_time).total_seconds()

                        if debug:
                            print('Start: {}'.format(current_stage_start_time))
                            print('Time (s): {}'.format(stage_time))
                            print()

                        recon_all_stages.append({'stage_name':stage_name, 'start':previous_stage_start_time, 'end':current_stage_start_time, 'time':stage_time})

            if debug:
                print('Short command:\n{}'.format(cmd_name.strip(' ')))
                print('Full command:\n{}'.format(cmd_line))
                print('Start Time:', start_time)
                print('End Time:', end_time)
                print('Total Time: {:.2f} seconds'.format(cmd_time))
                print()
                print('-'*30); print()

            if version == 1:
                string = temp_line + ' ' + start_time + ' ' + end_time + ' ' + str(cmd_time)
                if cmd_name == 'recon-all ':
                    string += '\n\nrecon-all stages:\n>>>\n'
                    for recon_all_stage in recon_all_stages:
                        string += recon_all_stage['stage_name'] + ' ' + recon_all_stage['start'] + ' ' + recon_all_stage['end'] + ' ' + str(recon_all_stage['time']) + '\n'
                    string += '<<<'
            elif version == 2:
                string = 'Command: {}\nStart: {}\nEnd: {}\nTime (s): {:.2f}'.format(temp_line, start_time,
                                                                                   end_time, cmd_time)
                if cmd_name == 'recon-all ':
                    string += '\n\nrecon-all stages:\n>>>'
                    for recon_all_stage in recon_all_stages:
                        string += '\nStage: {}\nStart: {}\nEnd: {}\nTime (s): {:.2f}\n'.format(recon_all_stage['stage_name'], recon_all_stage['start'],
                                                                                             recon_all_stage['end'], recon_all_stage['time'])
                    string += '<<<'

            output_file.write('\n'+string+'\n')

    output_file.close()
