import pandas as pd
import pathlib
import nanopore_event
from tqdm import tqdm
from typing import List
import numpy as np
import collections


def describe_dir(series: List) -> dict:
    pdseries = pd.Series(np.array(series))
    return dict(pdseries.describe())


def analyze(id: str, dir_path: pathlib.Path) -> dict:
    print(f"id: {id} start")
    result = {}
    result["id"] = id
    events_dir_path = dir_path.joinpath(id).joinpath("output/events")
    all_events = list(events_iter(events_dir_path))
    result["events_num"] = len(all_events)

    all_events = nanopore_event.AllEvent(all_events, Kmax=10)
    result["dwell_time"] = describe_dir(all_events.dwell_times())
    result["baseline"] = describe_dir(all_events.baselines())
    result["max_blockage"] = describe_dir(all_events.max_blockages())
    result["count_peak"] = dict(collections.Counter(all_events.peak_counts()))
    print(f"id: {id} done")
    return result


def events_iter(event_dir: pathlib.Path):
    return event_dir.iterdir()


def filter_events(event_filter, event_list: List[pathlib.Path]) -> List[pathlib.Path]:
    filtered_events = [
        event
        for event in tqdm(event_list, desc="Filtering")
        if catch_event_filter(event_filter, event)
    ]
    print(
        f"{len(filtered_events)}/{len(event_list)}: {int(len(filtered_events)/len(event_list)*100)}%"
    )
    return filtered_events


def dwell_time_filter(dt):
    return lambda event_path: nanopore_event.Event(event_path).dwell_time() > dt


def catch_event_filter(event_filter, event) -> bool:
    try:
        return event_filter(event)
    except:
        return False
