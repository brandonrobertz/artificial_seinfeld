#!/usr/bin/env python2.7
"""
Source: https://github.com/colinpollock/seinfeld-scripts
"""


import re
import sys



def unescape(s):
    """Replace HTML jibberish with normal symbols."""
    s = s.replace("&lt;", "<")
    s = s.replace("&gt;", ">")
    s = s.replace("&#145;", "'")
    s = s.replace("&#146;", "'")
    s = s.replace("&#147;", "'")
    s = s.replace("&#148;", "'")
    s = s.replace("&#149;", "'")
    s = s.replace("&#150;", "'")

    s = s.replace("&quot;", "'")
    s = s.replace("&#133;", "...")


    # this has to be last:
    s = s.replace("&amp;", "&")
    return s


def remove_tags(text):
    """Returns the text with HTML tags removed."""
    return re.sub(r'<.*?>', '', text)


def parse_episode_info(html):
    """Return a dict with meta-info about the episode."""
    groups = re.search(r'pc: .*? season (\d+), episode (\d+)', html).groups()
    season_num = int(groups[0])
    episode_num = int(groups[1])

    title = re.search(r'Episode \d+(.*?) - (.*?)<', html).groups()[1]
    date = re.search(r'Broadcast date: (.*?)<', html).groups()[0]
    writers = re.search(r'Written [bB]y:? (.*?)<', html).groups()[0]
    writers = tuple([w.strip() for w in re.split(r',|&amp;', writers) if w])
    director = re.search(r'Directed [bB]y (.*?)<', html).groups()[0]

    return {'season_num': season_num, 'episode_num': episode_num,
            'title': title, 'date': date, 'writers': writers,
            'director': director}


def parse_script(html):
    """Returns a sequence of (speaker, utterance) pairs."""
    utterances = re.findall(r'([A-Z]+)(?: \(.*?\))?: (.*?)</?(?:br|p)>', html)

    for i, (speaker, utterance_text) in enumerate(utterances):
        # Skip the monologues at the beginning of episodes.
        if speaker.upper() == 'JERRY' and \
              i == 0 and \
              len(utterance_text.split()) > 100:
            continue

        yield (speaker, unescape(utterance_text))



def scrape_episode(html):
    html = html.replace('&nbsp;', ' ')
    splitted = re.split(r'={30}.*', html)
    info_html = splitted[0]
    script_html = splitted[1]
    info = parse_episode_info(info_html)

    utterances = parse_script(script_html)
    return (info, utterances)

strip = r"\.\.\.|\.\.|\s'|'\s|\"|\(.*[\)\'\:]|\n|\r"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "USAGE: scrape.py script.shtml"
        sys.exit(1)
    with open(sys.argv[1]) as file:
        info, utterances = scrape_episode(file.read())
        last_line = None
        for utterance in utterances:
            speaker = utterance[0]
            line = re.sub( "\s{2,}", " ", re.sub( strip, " ", utterance[1])).lower()
            if last_line is None:
                last_line = line
                continue
            if speaker.lower() == 'jerry' \
                    and len(line) < 150 \
                    and len(last_line) < 150:
                print("{0}<Q>{1}<A>".format(
                    last_line, line
                ))
            last_line = line
