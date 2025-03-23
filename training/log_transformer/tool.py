import src.utils.env
from src.utils.re_exp2 import FormatMatcher
from src.utils.file import csv_read, log_read, csv_write

log_path = 'data/android/2k.log'
template_path = 'data/android/templates.csv'
log_structured_path = log_path + '_structured.csv'


templates = []
template_rows = csv_read(template_path)
template_rows.sort(key=lambda x: len(x['EventTemplate']), reverse=True) # sort by length of template
for row in template_rows:
    event_template = row['EventTemplate']
    event_template = event_template.replace('<*>', '%s')
    templates.append((event_template, ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']))
fm = FormatMatcher(templates)

log_rows, _ = log_read(log_path)
print('Total lines: ', len(log_rows))
# LineId,Date,Time,Pid,Tid,Level,Component,Content,EventId,EventTemplate,EventArgs
logs = []
error_line_num = 0
for line_id, row in enumerate(log_rows):
    event_id, args = fm.match(row['msg'])
    if event_id is not None:
        log = {}
        log['LineId'] = line_id + 1
        log['Date'] = row['date']
        log['Time'] = row['time']
        log['Pid'] = row['pid']
        log['Tid'] = row['tid']
        log['Level'] = row['level']
        log['Component'] = row['tag']
        log['Content'] = row['msg']
        log['EventId'] = template_rows[event_id]['EventId']
        log['EventTemplate'] = template_rows[event_id]['EventTemplate']
        log['EventArgs'] = ' | '.join(args.values())
        logs.append(log)
    else:
        error_line_num += 1
        # print('No match for line: ', line_id, row['msg'])

print('Error lines: ', error_line_num)
print('Matched lines: ', len(logs))

csv_write(log_structured_path, logs)
