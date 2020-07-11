"""
myutils.py

Util functions used throughout the project.
"""
import json
import os
import pickle
import sys
from timeit import default_timer as timer


# -----------------------------------------------------> SYSTEM <----------------------------------------------------- #

def make_folder(path):
    """ Create folder with init if not yet exists. """
    if not os.path.isdir(path):
        os.mkdir(path)
        with open(f'{path}/__init__.py', 'w') as f:
            f.write('')


# -----------------------------------------------------> SAVING <----------------------------------------------------- #


def save_as_file(obj: list, path: str, add_enter=True):
    """ Write file as raw list of strings. """
    with open(path, 'w', encoding="utf-8") as file:
        for row in obj:
            if add_enter:
                file.write("%s\n" % row)
            else:
                file.write("%s" % row)


def read_as_file(path: str):
    """ Read in a csv file. """
    with open(path, 'r', encoding="utf-8") as file:
        return file.readlines()
        # return list([item for row in csv.reader(file) for item in row])


def save_as_pickle(obj, path: str):
    """Write file as pickle object."""
    with open(f'{path}.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_as_pickle(path: str):
    """Read in pickle file."""
    with open(f'{path}.pickle', 'rb') as handle:
        return pickle.load(handle)


def save_as_json(obj, path: str, indent=2):
    """Write file as json object."""
    with open(f'{path}.json', 'w') as f:
        json.dump(obj, f, indent=indent)


def read_as_json(path: str):
    """Read in json object."""
    with open(f'{path}.json', 'r') as f:
        return json.load(f)


# -----------------------------------------------------> TIMING <----------------------------------------------------- #
time_dict = dict()


def drop(key=None, silent=False):
    """ Stop timing, print out duration since last preparation and append total duration. """
    # Update dictionary
    global time_dict
    if key not in time_dict:
        raise Exception("prep() must be summon first")
    time_dict[key]['end'] = timer()
    time_dict[None]['end'] = timer()
    
    # Determine difference
    start = time_dict[key]['start']
    end = time_dict[key]['end']
    diff = end - start
    
    # Fancy print diff
    if not silent:
        status_out(' done, ' + get_fancy_time(diff) + '\n')
    
    # Save total time
    if key is not None:
        if 'sum' not in time_dict[key]:
            time_dict[key]['sum'] = diff
        else:
            time_dict[key]['sum'] += diff
    
    return diff


def get_fancy_time(sec):
    """
    Convert a time measured in seconds to a fancy-printed time.

    :param sec: Float
    :return: String
    """
    h = int(sec) // 3600
    m = (int(sec) // 60) % 60
    s = sec % 60
    if h > 0:
        return '{h} hours, {m} minutes, and {s} seconds.'.format(h=h, m=m, s=round(s, 2))
    elif m > 0:
        return '{m} minutes, and {s} seconds.'.format(m=m, s=round(s, 2))
    else:
        return '{s} seconds.'.format(s=round(s, 2))


def intermediate(key=None):
    """ Get the time that already has passed. """
    # Update dictionary
    global time_dict
    if key not in time_dict:
        raise Exception("prep() must be summon first")
    
    # Determine difference
    start = time_dict[key]['start']
    end = timer()
    return end - start


def prep(msg="Start timing...", key=None, silent=False):
    """ Prepare timing, print out the given message. """
    global time_dict
    if key not in time_dict:
        time_dict[key] = dict()
    if not silent:
        status_out(msg)
    time_dict[key]['start'] = timer()
    
    # Also create a None-instance (in case drop() is incorrect)
    if key:
        if None not in time_dict:
            time_dict[None] = dict()
        time_dict[None]['start'] = timer()


def print_all_stats():
    """ Print out each key and its total (cumulative) time. """
    global time_dict
    if time_dict:
        if None in time_dict: del time_dict[None]  # Remove None-instance first
        print("\n\n\n---------> OVERVIEW OF CALCULATION TIME <---------\n")
        keys_space = max(map(lambda x: len(x), time_dict.keys()))
        line = ' {0:^' + str(keys_space) + 's} - {1:^s}'
        line = line.format('Keys', 'Total time')
        print(line)
        print("-" * (len(line) + 3))
        line = '>{0:^' + str(keys_space) + 's} - {1:^s}'
        t = 0
        for k, v in sorted(time_dict.items()):
            try:
                t += v['sum']
                print(line.format(k, get_fancy_time(v['sum'])))
            except KeyError:
                print("KeyError: Probably you forgot to place a 'drop()' in the {} section".format(k))
                raise KeyError
        end_line = line.format('Total time', get_fancy_time(t))
        print("-" * (len(end_line)))
        print(end_line)


def remove_time_key(key: str):
    """
    Remove a key from the timing-dictionary.

    :param key: Key that must be removed
    """
    global time_dict
    if key in time_dict:
        del time_dict[key]


def status_out(msg):
    """ Write the given message. """
    sys.stdout.write(msg)
    sys.stdout.flush()
