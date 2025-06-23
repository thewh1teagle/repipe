import numpy as np
from numpy.typing import NDArray
import json
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader
import onnxruntime as ort


_BOS = "^"
_EOS = "$"
_PAD = "_"


class Piper:
    def __init__(
            self, 
            model_path: str, 
            config_path: str,
        ):
        self.setup(model_path, config_path)
        
    def setup(self, model_path, config_path, session = None):
        with open(config_path) as fp:
            self.config: dict = json.load(fp)
        self.sample_rate: int = self.config['audio']['sample_rate']
        self.phoneme_id_map: dict = self.config['phoneme_id_map']
        self._voices: dict = self.config.get('speaker_id_map')

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        self.sess = session or ort.InferenceSession(
            model_path, 
            sess_options=ort.SessionOptions(),
            providers=['CPUExecutionProvider']
        )
        self.sess_inputs_names = [i.name for i in self.sess.get_inputs()]

    @classmethod
    def from_session(
        cls,
        session: ort.InferenceSession,
        config_path: str,
    ):
        instance = cls.__new__(cls)
        instance.setup(model_path='', config_path=config_path, session=session)
        return instance

    def create(
            self, 
            text: str, 
            speaker_id: str | int = None, 
            is_phonemes = False,
            length_scale: int = None,
            noise_scale: int = None,
            noise_w: int = None,
        ) -> tuple[NDArray[np.float32], int]:

        inference_cfg = self.config['inference']
        length_scale = length_scale or inference_cfg['length_scale']
        noise_scale = noise_scale or inference_cfg['noise_scale']
        noise_w = noise_w or inference_cfg['noise_w']

        sid = 0
        if isinstance(speaker_id, str) and speaker_id in self._voices:
            sid = self._voices[speaker_id]
        elif isinstance(speaker_id, int):
            sid = speaker_id
        
        phonemes = text if is_phonemes else phonemize(text)
        phonemes = list(phonemes)
        phonemes.insert(0, _BOS)

        ids = self._phoneme_to_ids(phonemes)
        
        inputs = self._create_input(ids, length_scale, noise_w, noise_scale, sid)

        samples = self.sess.run(None, inputs)[0].squeeze((0,1)).squeeze()
        return samples, self.sample_rate
    
    def get_voices(self) -> dict | None:
        return self._voices

    def _phoneme_to_ids(self, phonemes: str) -> list[int]:
        ids = []
        for p in phonemes:
            if p in self.phoneme_id_map:
                ids.extend(self.phoneme_id_map[p])
                ids.extend(self.phoneme_id_map[_PAD])
        ids.extend(self.phoneme_id_map[_EOS])
        return ids
    
    def _create_input(self, ids: list[int], length_scale: int, noise_w: int, noise_scale: int, sid: int) -> dict:
        ids = np.expand_dims(np.array(ids, dtype=np.int64), 0)
        length = np.array([ids.shape[1]], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w],dtype=np.float32)
        
        sid = np.array([sid], dtype=np.int64) if sid is not None else None
        input = {
            'input': ids,
            'input_lengths': length,
            'scales': scales,
        }
        if 'sid' in self.sess_inputs_names:
            input['sid'] = sid
        return input