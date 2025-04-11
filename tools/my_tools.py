from datetime import datetime

import pyreadr


def get_filename(time):
    return f'../data/Dresden_01-03.24/{time.strftime("%Y-%m-%d")}/{time.strftime("%Y-%m-%d-%H-%M-00")}.rds'


def import_raw_file(year, month, day, hour, minute, second):
    time = datetime(year, month, day, hour, minute, second)
    current_filename = get_filename(time)
    df = pyreadr.read_r(current_filename)[None]
    return df
