import typer
import json
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

from google_play_scraper import Sort, reviews, app

sns.set(style="whitegrid", palette="muted", font_scale=1.2)


def print_json(json_object):
    json_str = json.dumps(json_object, indent=2, sort_keys=True, default=str)
    print(highlight(json_str, JsonLexer(), TerminalFormatter()))


def scrape(apps_dataset_path, reviews_datase_path, show_debug: bool = False):
    app_packages = [
        "com.anydo",
        "com.todoist",
        "com.ticktick.task",
        "com.habitrpg.android.habitica",
        "cc.forestapp",
        "com.oristats.habitbull",
        "com.levor.liferpgtasks",
        "com.habitnow",
        "com.microsoft.todos",
        "prox.lab.calclock",
        "com.gmail.jmartindev.timetune",
        "com.artfulagenda.app",
        "com.tasks.android",
        "com.appgenix.bizcal",
        "com.appxy.planner",
    ]

    app_infos = []

    for ap in tqdm(app_packages):
        info = app(ap, lang="en", country="us")
        del info["comments"]
        app_infos.append(info)

    if show_debug:
        print_json(app_infos[0])

    app_infos_df = pd.DataFrame(app_infos)

    if show_debug:
        print(app_infos_df)

    app_infos_df.to_csv(apps_dataset_path, index=None, header=True)

    app_reviews = []

    for ap in tqdm(app_packages):
        for score in range(1, 6):
            for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
                rvs, _ = reviews(
                    ap,
                    lang="en",
                    country="us",
                    sort=sort_order,
                    count=300 if score == 3 else 150,
                    filter_score_with=score,
                )
                for r in rvs:
                    r["sortOrder"] = (
                        "most_relevant"
                        if sort_order == Sort.MOST_RELEVANT
                        else "newest"
                    )
                    r["appId"] = ap
                app_reviews.extend(rvs)

    if show_debug:
        print(app_reviews[0])

    app_reviews_df = pd.DataFrame(app_reviews)
    app_reviews_df.to_csv(reviews_datase_path, index=None, header=True)


if __name__ == "__main__":
    typer.run(scrape)