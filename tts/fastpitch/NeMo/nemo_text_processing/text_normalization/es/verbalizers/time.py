# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pynini
from NeMo.nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)
from NeMo.nemo_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

alt_minutes = pynini.string_file(get_abs_path("data/time/alt_minutes.tsv"))

morning_times = pynini.string_file(get_abs_path("data/time/morning_times.tsv"))
afternoon_times = pynini.string_file(get_abs_path("data/time/afternoon_times.tsv"))
evening_times = pynini.string_file(get_abs_path("data/time/evening_times.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "doce" minutes: "media" suffix: "a m" } -> doce y media de la noche
        time { hours: "doce" } -> twelve o'clock

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        change_minutes = pynini.cdrewrite(alt_minutes, pynini.accep("[BOS]"), pynini.accep("[EOS]"), NEMO_SIGMA)

        morning_phrases = pynini.cross("am", "del mattino")
        afternoon_phrases = pynini.cross("pm", "del pomeriggio")
        evening_phrases = pynini.cross("pm", "di notte")

        # For the 12's
        mid_times = pynini.accep("doce")
        mid_phrases = (
            pynini.string_map([("pm", "del mezzogiorno"), ("am", "di notte")])
            if deterministic
            else pynini.string_map(
                [
                    ("pm", "del mattino"),
                    ("pm", "del giorno"),
                    ("pm", "del mezzogiorno"),
                    ("am", "di notte"),
                    ("am", "della mezzanotte"),
                ]
            )
        )

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        minute = (minute @ change_minutes) if deterministic else pynini.union(minute, minute @ change_minutes)

        suffix = (
            pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        zone = (
            pynutil.delete("zone:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_zone = pynini.closure(delete_space + insert_space + zone, 0, 1)
        second = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        graph_hms = (
            hour
            + pynutil.insert(" ore ")
            + delete_space
            + minute
            + pynutil.insert(" minuti e ")
            + delete_space
            + second
            + pynutil.insert(" secondi")
        )

        graph_hm = hour + delete_space + pynutil.insert(" e ") + minute
        graph_hm |= pynini.union(
            (hour @ morning_times)
            + delete_space
            + pynutil.insert(" e ")
            + minute
            + delete_space
            + insert_space
            + (suffix @ morning_phrases),
            (hour @ afternoon_times)
            + delete_space
            + pynutil.insert(" e ")
            + minute
            + delete_space
            + insert_space
            + (suffix @ afternoon_phrases),
            (hour @ evening_times)
            + delete_space
            + pynutil.insert(" e ")
            + minute
            + delete_space
            + insert_space
            + (suffix @ evening_phrases),
            (hour @ mid_times)
            + delete_space
            + pynutil.insert(" e ")
            + minute
            + delete_space
            + insert_space
            + (suffix @ mid_phrases),
        )

        graph_h = pynini.union(
            hour,
            (hour @ morning_times) + delete_space + insert_space + (suffix @ morning_phrases),
            (hour @ afternoon_times) + delete_space + insert_space + (suffix @ afternoon_phrases),
            (hour @ evening_times) + delete_space + insert_space + (suffix @ evening_phrases),
            (hour @ mid_times) + delete_space + insert_space + (suffix @ mid_phrases),
        )

        graph = (graph_hms | graph_hm | graph_h) + optional_zone

        if not deterministic:
            graph_style_1 = pynutil.delete(" style: \"1\"")
            graph_style_2 = pynutil.delete(" style: \"2\"")

            graph_menos = hour + delete_space + pynutil.insert(" meno ") + minute + graph_style_1
            graph_menos |= (
                (hour @ morning_times)
                + delete_space
                + pynutil.insert(" meno ")
                + minute
                + delete_space
                + insert_space
                + (suffix @ morning_phrases)
                + graph_style_1
            )
            graph_menos |= (
                (hour @ afternoon_times)
                + delete_space
                + pynutil.insert(" meno ")
                + minute
                + delete_space
                + insert_space
                + (suffix @ afternoon_phrases)
                + graph_style_1
            )
            graph_menos |= (
                (hour @ evening_times)
                + delete_space
                + pynutil.insert(" meno ")
                + minute
                + delete_space
                + insert_space
                + (suffix @ evening_phrases)
                + graph_style_1
            )
            graph_menos |= (
                (hour @ mid_times)
                + delete_space
                + pynutil.insert(" meno ")
                + minute
                + delete_space
                + insert_space
                + (suffix @ mid_phrases)
                + graph_style_1
            )
            graph_menos += optional_zone

            graph_para = minute + pynutil.insert(" per le ") + delete_space + hour + graph_style_2
            graph_para |= (
                minute
                + pynutil.insert(" per le ")
                + delete_space
                + (hour @ morning_times)
                + delete_space
                + insert_space
                + (suffix @ morning_phrases)
                + graph_style_2
            )
            graph_para |= (
                minute
                + pynutil.insert(" per le ")
                + delete_space
                + (hour @ afternoon_times)
                + delete_space
                + insert_space
                + (suffix @ afternoon_phrases)
                + graph_style_2
            )
            graph_para |= (
                minute
                + pynutil.insert(" per le ")
                + delete_space
                + (hour @ evening_times)
                + delete_space
                + insert_space
                + (suffix @ evening_phrases)
                + graph_style_2
            )
            graph_para |= (
                minute
                + pynutil.insert(" per le ")
                + delete_space
                + (hour @ mid_times)
                + delete_space
                + insert_space
                + (suffix @ mid_phrases)
                + graph_style_2
            )
            graph_para += optional_zone
            graph_para @= pynini.cdrewrite(
                pynini.cross(" le ", " la "), "per", "una", NEMO_SIGMA
            )  # Need agreement with one

            graph |= graph_menos | graph_para
        delete_tokens = self.delete_tokens(graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
