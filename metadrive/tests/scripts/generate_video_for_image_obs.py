import distutils.spawn
import distutils.version
import os
import subprocess
import time

import numpy as np
from gym import logger, error
from panda3d.core import PNMImage

from metadrive.component.algorithm.BIG import BigGenerateMethod
from metadrive.component.map.base_map import BaseMap
from metadrive.envs.metadrive_env import MetaDriveEnv


class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise error.InvalidFrame(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), "
                "i.e., RGB values for a w-by-h image, with an optional alpha "
                "channel.".format(frame_shape)
            )
        self.wh = (w, h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        elif distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise error.DependencyNotInstalled(
                """Found neither the ffmpeg nor avconv executables. On OS X, 
                you can install ffmpeg via `brew install ffmpeg`. On most 
                Ubuntu variants, `sudo apt-get install ffmpeg` should do it. 
                On Ubuntu 14.04, however, you'll need to install avconv with 
                `sudo apt-get install libav-tools`."""
            )

        self.start()

    @property
    def version_info(self):
        return {
            'backend': self.backend,
            'version': str(subprocess.check_output([self.backend, '-version'], stderr=subprocess.STDOUT)),
            'cmdline': self.cmdline
        }

    def start(self):
        self.cmdline = (
            self.backend,
            '-nostats',
            '-loglevel',
            'error',  # suppress warnings
            '-y',
            '-r',
            '%d' % self.frames_per_sec,
            # '-b', '2M',

            # input
            '-f',
            'rawvideo',
            '-s:v',
            '{}x{}'.format(*self.wh),
            '-pix_fmt',
            ('rgb32' if self.includes_alpha else 'rgb24'),
            '-i',
            '-',
            # this used to be /dev/stdin, which is not Windows-friendly

            # output
            '-vf',
            'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-vcodec',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-crf',
            '18',
            # '-vtag',
            # 'hvc1',
            self.output_path
        )

        logger.debug('Starting ffmpeg with "%s"', ' '.join(self.cmdline))
        if hasattr(os, 'setsid'):  # setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame(
                'Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame)
            )
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame(
                "Your frame has shape {}, but the VideoRecorder is "
                "configured for shape {}.".format(frame.shape, self.frame_shape)
            )
        if frame.dtype != np.uint8:
            raise error.InvalidFrame(
                "Your frame has data type {}, but we require uint8 (i.e. RGB "
                "values from 0-255).".format(frame.dtype)
            )

        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error("VideoRecorder encoder exited with status {}".format(ret))


def gen_video(frames, file="tmp"):
    path = "{}.mp4".format(file)
    encoder = ImageEncoder(path, frames[0].shape, 50)
    for f in frames:
        encoder.capture_frame(f)
    encoder.close()
    del encoder
    print("Video is saved at: ", path)
    # video = io.open(path, 'r+b').read()
    # encoded = base64.b64encode(video)
    # ipythondisplay.display(HTML(data='''<video alt="test" autoplay
    #           loop controls style="height: 400px;">
    #           <source src="data:video/mp4;base64,{0}" type="video/mp4" />
    #         </video>'''.format(encoded.decode('ascii'))))


if __name__ == '__main__':
    headless = True
    env = MetaDriveEnv(
        dict(
            use_render=False,
            map_config={
                BaseMap.GENERATE_TYPE: BigGenerateMethod.BLOCK_NUM,
                BaseMap.GENERATE_CONFIG: 7
            },
            traffic_density=0.5,
            offscreen_render=True,
            headless_machine_render=headless
        )
    )
    start = time.time()
    env.reset()
    frames = []
    for num_frames in range(30):
        o, r, d, info = env.step([0, 1])
        img = PNMImage()
        env.engine.win.getScreenshot(img)
        frame = np.zeros([1200, 900, 4], dtype=np.uint8)
        for i in range(1200):
            for j in range(900):
                frame[i, j] = img.get_pixel(i, j)
        frame = frame.swapaxes(0, 1)[..., :3]
        frames.append(frame)
        print(f"Finish {num_frames + 1} frames")
    env.close()
    gen_video(frames)
