from abc import ABC, abstractmethod
import numpy as np
from logger import get_logger
import requests
from beanie.sync import init_beanie
from pymongo import MongoClient
from schemas_sync import *
from podcast import PodcastGenerator
from schemas_sync import *
from enums import *
from fastpitch.tts_model import FastpitchModel
from vits.tts_model import VitsModel
from synthesizer import Synthesizer
import os
from pytz import timezone
from podcast import PodcastGenerator
import os
import torch
import time
from pydub import AudioSegment
from article_scraper import scrape_article

log = get_logger(__name__)


class Worker(ABC):
    """
    Base Worker class. Every TTS worker has to inherit from it.
    """

    def __init__(self):
        os.makedirs(f"/files/articles", exist_ok=True)
        os.makedirs(f"/files/podcasts", exist_ok=True)
        # init db session
        try:
            self.client = MongoClient(
                "mongodb://mongo:27017"
            )
        except pymongo.errors.ConnectionFailure as cf:
            log.fatal(f"Could not connect to database: {cf}")
            exit(1)

        init_beanie(database=self.client.podcast_db,
                    document_models=[Podcast])
        log.info("Connected to database.")
        self.podcast = PodcastGenerator()

    def get_articles(self, article_urls):
        """Retrieves articles given the urls.

        Parameters
        ----------
        article_urls : list of str
            list of article urls

        Returns
        -------
        list of dict
            a list of {"url": <article_url>, "text": <fulltext.>}
        """
        articles = list(dict())
        for url in article_urls:
            article = { "url": url, "text": scrape_article(url) }
            if article:
                articles.append(article)
            else:
                log.error(f"Could not scrape article: {url}")
                return None

        return articles

    def save_as_mp3(self, audio_path: str, audio: np.ndarray):
        """Saves np.array audio wave as mp3

        Parameters
        ----------
        audio_path : str
            audio file path
        audio : np.array
            audio waveform
        """
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=16000,
            sample_width=audio.dtype.itemsize,
            channels=1
        )
        audio_segment.export(audio_path, format="mp3", bitrate="160k")

    @abstractmethod
    def run_inference(self, ch, method, properties, body):
        """Callback called by pika (RabbitMQ client).
        Consumes messages from the task queue and generates podcasts.

        Parameters
        ----------
        body : byte str
            the byte encoded string containing the podcast_id
        """
        pass


class FastPitchWorker(Worker):
    def __init__(self, n_torch_threads):
        super().__init__()
        torch.set_num_threads(n_torch_threads)
        BASE = "/checkpoints/fastpitch"
        self.male1_model = FastpitchModel(
            f"{BASE}/male1/FastPitch.ckpt", f"{BASE}/male1/HifiGan.ckpt", "./male_conf.yaml")
        self.female1_model = FastpitchModel(
            f"{BASE}/female1/FastPitch.ckpt", f"{BASE}/female1/HifiGan.ckpt", "./female_conf.yaml")

        self.male1_synthesizer = Synthesizer(self.male1_model)
        self.female1_synthesizer = Synthesizer(self.female1_model)

    def run_inference(self, ch, method, properties, body):
        podcast_id = body.decode()
        log.debug(f"Received podcast: {podcast_id}")
        podcast = ~Podcast.get(podcast_id)
        podcast.update({"$set": {Podcast.status: Status.Running}})

        if podcast.voice == Voice.Female1:
            synthesizer = self.female1_synthesizer
        elif podcast.voice == Voice.Male1:
            synthesizer = self.male1_synthesizer

        audios = []
        articles = super().get_articles(podcast.article_urls)
        if articles:
            for article in articles:
                article_name = article["url"].split("/")[-1]
                audio_path = f"/files/articles/{article_name}_{podcast.voice}.mp3"

                if not os.path.isfile(audio_path):
                    text = article["text"]
                    log.debug(f"Synthesizing article: {article['url']}")
                    audio = synthesizer.text_to_speech(text)
                    audio = super().save_as_mp3(audio_path, audio)
                else:
                    log.debug(
                        f"Article: {article['url']} has already been generated! Skipping inference.")

                audio_segment = AudioSegment.from_mp3(audio_path)
                audios.append(audio_segment)

            final_segment = self.podcast.generate_segment(audios)
            podcast_path = f"/files/podcasts/{podcast_id}.mp3"
            final_segment.export(podcast_path, format="mp3", bitrate="160k")
            log.debug(f"Successfully generated podcast: {podcast_id}")
            podcast.update({"$set": {Podcast.status: Status.Succeeded,
                            Podcast.file_path: podcast_path,
                                     Podcast.created_at: datetime.now(timezone("Europe/Rome"))}})

        else:  # request failed
            log.error(f"Failed at generating podcast: {podcast_id}")
            podcast.update({"$set": {Podcast.status: Status.Failed}})
        ch.basic_ack(delivery_tag=method.delivery_tag)


class VitsWorker(Worker):
    def __init__(self, n_torch_threads):
        super().__init__()
        torch.set_num_threads(n_torch_threads)
        BASE = "/checkpoints/vits"
        self.vits_model = VitsModel(f"{BASE}/vits.pth")
        self.synthesizer = Synthesizer(self.vits_model)

    def run_inference(self, ch, method, properties, body):
        podcast_id = body.decode()
        log.debug(f"Received podcast: {podcast_id}")
        podcast = ~Podcast.get(podcast_id)
        podcast.update({"$set": {Podcast.status: Status.Running}})

        audios = []
        articles = super().get_articles(podcast.article_urls)
        if articles:
            for article in articles:
                article_name = article["url"].split("/")[-1]
                audio_path = f"/files/articles/{article_name}_vits.mp3"

                if not os.path.isfile(audio_path):
                    text = article["text"]
                    log.debug(f"Synthesizing article: {article['url']}")
                    audio = self.synthesizer.text_to_speech(text)
                    audio = super().save_as_mp3(audio_path, audio)
                else:
                    log.debug(
                        f"Article: {article['url']} has already been generated! Skipping inference.")

                audio_segment = AudioSegment.from_mp3(audio_path)
                audios.append(audio_segment)

            final_segment = self.podcast.generate_segment(audios)
            podcast_path = f"/files/podcasts/{podcast_id}.mp3"
            final_segment.export(podcast_path, format="mp3", bitrate="160k")
            log.debug(f"Successfully generated podcast: {podcast_id}")
            podcast.update({"$set": {Podcast.status: Status.Succeeded,
                            Podcast.file_path: podcast_path,
                                     Podcast.created_at: datetime.now(timezone("Europe/Rome"))}})

        else:  # request failed
            log.error(f"Failed at generating podcast: {podcast_id}")
            podcast.update({"$set": {Podcast.status: Status.Failed}})
        ch.basic_ack(delivery_tag=method.delivery_tag)

