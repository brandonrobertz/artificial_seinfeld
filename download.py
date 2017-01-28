#!/usr/bin/env python2.7
"""Script for downloading Seinfeld scripts.
Source: https://github.com/colinpollock/seinfeld-scripts
"""

from __future__ import division
import argparse
import os
import sys
import time

import requests


def _format_num(n):
    return '%02d' % n

EPISODE_NUMBERS = (
    map(_format_num, range(1, 82)) +

    # Double episode
    ['82and83'] +

    map(_format_num, range(84, 100)) +

    # Skip the clip show "100and101".

    map(_format_num, range(102, 177)) +

    # Skip the clip show "177and178".

    # Double episode (Finale)
    ['179and180']
)

SCRIPT_URL = 'http://www.seinology.com/scripts/script-%s.shtml'
SUMMARY_URL = 'http://www.seinology.com/epguide/%s.shtml'

# how many times to retry downloading in case of temporary network/server issue
RETRIES = 3


def get_html(episode_number, summary=False):
    URL = SCRIPT_URL if not summary else SUMMARY_URL
    url = URL % episode_number
    print "Fetching", url
    tries = 0
    while True:
        try:
            resp = requests.get(url)
            if resp.status_code == 404:
                print("Got 404, skipping")
                return None
            resp.raise_for_status()
        except Exception as e:
            print("Error: {0}".format(e))
            if tries < RETRIES:
                print("Retrying {0} more times ...". format(RETRIES - tries))
                tries += 1
                continue
            else:
                print("Max retries. Skipping.")
                return None
        return resp.text


def main(args):
    episode_numbers = (
        args.episode_number if args.episode_number is not None
        else EPISODE_NUMBERS
    )

    num_episodes = len(episode_numbers)
    for idx, episode_number in enumerate(episode_numbers, start=1):
        print "Downloading EP", episode_number
        print >> sys.stderr, '[Ep. %s]\t\t%d of %d (%.2f%%)' % (
            episode_number, idx,
            len(episode_numbers),
            idx / num_episodes * 100
        )

        script_out_path = os.path.join(
            args.output_directory,
            '%s.shtml' % episode_number
        )

        if args.no_overwrite is True and os.path.exists(script_out_path):
            continue

        html = get_html(episode_number, summary=bool(args.summary))

        if html is not None:
            with open(script_out_path, 'w') as fh:
                print >> fh, html

        if idx != num_episodes:
            time.sleep(args.sleep_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'output_directory',
        help='The directory to write script HTML files to.'
    )

    parser.add_argument(
        '--sleep-seconds',
        type=float,
        default=1.0,
        help=('The number of seconds to sleep between downloading each page. '
              'Defaults to 1 second.')
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help=('Download script summaries instead of transscript. NOTE: You '
              'may wish to store this in a new directory, as it will '
              'ignore/overwrite any scripts in the given directory without '
              'looking to see if it is a summary or transcript.')
    )

    parser.add_argument(
        '--no-overwrite',
        action='store_false',
        default=True,
        help='If True then will not overwrite existing downloaded files.'
    )

    parser.add_argument(
        '--episode-number',
        action='append',
        required=False,
        choices=EPISODE_NUMBERS,
        help=('An episode name or number (e.g. "79" or "82and83"). If '
              'specified then only this episode will be downloaded '
              'rather than all of them (the default). This can be specified '
              'multiple times.')
    )

    main(parser.parse_args(sys.argv[1:]))
