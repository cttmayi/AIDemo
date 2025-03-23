import csv
import re


PARSER_ANDROID_REGEX = r'(\d+-\d+) (\d+:\d+:\d+\.\d+) +(\d+) +(\d+) ([A-Z]) (.+?): (.*)'
# 01-27 23:53:12.299   665 12258 W ratelimit: Single process limit 50/s drop 280 lines.
# 11-25 19:41:19.813  1153  1153 F libc    : Fatal signal 6 (SIGABRT), code -1 (SI_QUEUE) in tid 1153 (init), pid 1153 (init)
# 03-17 16:13:38.859  2227  2227 D TextView: visible is system.time.showampm
def android_parser(lines):
    regex = PARSER_ANDROID_REGEX
    rows = []
    pattern = re.compile(regex)
    for line in lines:
        match = pattern.match(line)
        if match:
            row = {'date': match.group(1), 'time': match.group(2), 'pid': match.group(3), 'tid': match.group(4), 'level': match.group(5), 'tag': match.group(6), 'msg': match.group(7)}
            rows.append(row)
    return rows



AUTO_PARSER = {
    PARSER_ANDROID_REGEX: {
        'parser': android_parser,
        'type': 'android'
    },
}

def auto_parser(lines, match_count=3):
    for regex, parser in AUTO_PARSER.items():
        pattern = re.compile(regex)
        count = 0
        for line in lines:
            match = pattern.match(line)
            if match:
                count += 1
        if count > match_count:
            return parser['parser'](lines), parser['type']

    return None, None
