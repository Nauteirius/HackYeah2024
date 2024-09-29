import io
import os
from typing import NamedTuple
from tempfile import NamedTemporaryFile

import ffmpeg
import numpy as np


class AudioVideo(NamedTuple):
    fps: int
    width: int
    height: int
    frames: list[np.ndarray]
    audio: io.BytesIO


def split(video_bytes: io.BytesIO) -> AudioVideo:

    isWindows = os.name == 'nt'
    # workaround JUST for windows NT :)

    with NamedTemporaryFile(mode='w+b', delete=(not isWindows)) as tmp_input:
        tmp_input.write(video_bytes.read())

        file_name = tmp_input.file.name

        probe = ffmpeg.probe(file_name)
        probe_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        # TODO: somehow propagate errors if needed
        fps = int(probe_stream['avg_frame_rate'].split('/')[0])
        width = int(probe_stream['width'])
        height = int(probe_stream['height'])

        out, _ = (
            ffmpeg.input(file_name)
            .video.output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run(capture_stdout=True)
        )
        frames_4d = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        frames = [
            arr.squeeze(0) for arr in np.split(frames_4d, frames_4d.shape[0], axis=0)
        ]

        out, _ = (
            ffmpeg.input(file_name)
            .audio.output('pipe:', format='mp3', acodec='libmp3lame')
            .run(capture_stdout=True)
        )

        mp3_bytes = io.BytesIO()
        mp3_bytes.name = file_name + '.mp3'
        mp3_bytes.write(out)
        mp3_bytes.seek(0)
    
    # workaround JUST for windows NT :)
    if isWindows and file_name:
        os.remove(file_name)

    return AudioVideo(fps, width, height, frames, mp3_bytes)
