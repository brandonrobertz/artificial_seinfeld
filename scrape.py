#!/usr/bin/env python2.7
"""
Source: https://github.com/colinpollock/seinfeld-scripts
"""
from __future__ import print_function
import lxml.html
import re
import argparse

end_q_sequence = '|'
end_a_sequence = '#'

# this is responsible for cleaning the scripts
strip = "\.\.\.|\.\.|\s'|'\s|\"|\(.*[\)\'\:]|\n|\r|\d|\*.*\*|" + \
    "[^A-Za-z\'\.\?\! ]"


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


def scrape_transcript(html):
    html = html.replace('&nbsp;', ' ')
    splitted = re.split(r'={30}.*', html)
    info_html = splitted[0]
    script_html = splitted[1]
    info = parse_episode_info(info_html)

    utterances = parse_script(script_html)
    return (info, utterances)


def scrape_summary(filename):
    tree = lxml.html.parse(open(filename))
    path = tree.xpath("//*[text()='Summary']/../../../../div[3]/p/text()")
    description = (' ').join(path)
    title_raw = (' ').join(tree.xpath('/html/head/title/text()'))
    title_lower = title_raw.split('::')[-1].strip()
    title = re.sub(strip, " ", title_lower)
    return (title, description)


def clean(data):
    return re.sub(
        "\s{2,}",
        " ",
        re.sub(strip, " ", data)
    ).lower().strip()


def args():
    desc = 'Take a raw SHTML Seinfeld transcript, downloaded using '\
        'download.py from seinology.com, and extract statement ' \
        'and response pair for a given Seinfeld character.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('script', type=str,
                        help='Seinology SHTML transcript file')
    parser.add_argument('--mode', default='character',
                        choices=['character','synopsis'],
                        help='Mode to run scraper in. There are currently ' \
                        'two modes: character and synopsis. Character mode ' \
                        'extracts character response/answer pairs, for the ' \
                        'purpose of training a character model. Synopsis ' \
                        'mode extracts title/synopsis pairs, intended to ' \
                        'be used to take a title (Seinfeld titles are ' \
                        'typically nouns or specific actions) and generate ' \
                        'a synopsis from it.')
    parser.add_argument('--character', default='jerry',
                        help='Seinfeld script character to pull responses ' \
                        'or "all" for all character dialogue. Character mode ' \
                        'only.')
    args = parser.parse_args()
    return args.script, args.character, args.mode


if __name__ == "__main__":
    path, character, mode = args()
    if mode == 'character':
        file = open(path)
        info, utterances = scrape_transcript(file.read())
        last_line = None
        for utterance in utterances:
            speaker = utterance[0].lower().strip()
            line = clean(utterance[1])
            if last_line is None:
                last_line = line
                continue
            if character == "all" or speaker == character \
                    and len(line) < 150 \
                    and len(line) > 15 \
                    and len(last_line) < 150 \
                    and len(last_line) > 15:
                print("{0}{1}{2}{3}".format(
                    last_line.lower(),
                    end_q_sequence,
                    line.lower(),
                    end_a_sequence
                ))
            last_line = line
        file.close()
    elif mode == 'synopsis':
        title, synopsis = scrape_summary(path)
        title = clean(title)
        synopsis = clean(synopsis)
        print("{0}{1}{2}{3}".format(
            title,
            end_q_sequence,
            synopsis,
            end_a_sequence
        ))

