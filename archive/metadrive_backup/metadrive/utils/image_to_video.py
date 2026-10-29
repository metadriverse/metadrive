import os

import cv2
from tqdm.auto import tqdm


def image_files_to_video(video_name, image_folder, code="mp4v"):
    """
    code=mp4v, avc1, x264, h264 etc.
    """
    assert video_name.endswith(".mp4")
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x[:-4]))
    assert len(images) > 0
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*code), 40, (width, height))
    for image in tqdm(images, desc="Writing Video"):
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error happen: ", e)


def image_list_to_video(video_name, image_list, code="mp4v"):
    """
    code=mp4v, avc1, x264, h264 etc.
    """
    assert video_name.endswith(".mp4")
    assert len(image_list) > 0
    # frame = cv2.imread(os.path.join(image_folder, images[0]))
    frame = image_list[0]
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*code), 40, (width, height))
    for image in tqdm(image_list, desc="Writing Video"):

        # Change color
        image = image[..., ::-1]

        video.write(image)
    video.release()
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error happen: ", e)
