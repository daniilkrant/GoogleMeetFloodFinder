import sched
import time
import datetime

import cv2
from fuzzywuzzy import fuzz
import numpy as np
from mss import mss
from pytesseract import pytesseract

stats = dict()


def get_stats():
    format_stats()
    print_stats()


def format_stats():
    speakers = list(stats.keys())
    for i in range(len(speakers)):
        if speakers[i] in "pass":
            continue
        for j in range(i, len(speakers)):
            ratio = fuzz.ratio(speakers[i], speakers[j])
            # print(f"Ratio for {speakers[i]}, {speakers[j]} is {ratio}")
            if 80 < ratio < 100:
                stats[speakers[i]] += stats[speakers[j]]
                stats.pop(speakers[j])
                speakers[j] = "pass"


def print_stats():
    print("------------------------------------------------------------")
    for speaker in stats:
        print(f"Speaker: {speaker}, time: {str(datetime.timedelta(seconds=stats[speaker]))}")
    print("------------------------------------------------------------")


def current_milli_time():
    return round(time.time() * 1000)


def get_next_image():
    with mss() as sct:
        monitor = sct.monitors[2]
        screenshot = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2RGBA)
        return screenshot
    # return cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_files/screenshot_eng.jpg'))


def find_speaker_contour(image_to_process):
    edged = cv2.Canny(image_to_process, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_contour = max(contours, key=cv2.contourArea)
    return biggest_contour


def crop_by_contour(contour, image_to_process):
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image_to_process[y:y + h, x:x + w]
    return cropped


def show(image_to_process, title):
    cv2.imshow(title, image_to_process)


def prepare_for_ocr(active_user_image):
    active_user_image = cv2.cvtColor(active_user_image, cv2.COLOR_BGR2GRAY)
    active_user_image = cv2.threshold(active_user_image, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    x, y, w, h = cv2.boundingRect(active_user_image)
    cropped = active_user_image[round(h * 0.8):h, round(w * 0.01):round(w * 0.5)]
    return cropped


def get_speaker_name(image):
    custom_config = r'--oem 3 --psm 7'
    res = pytesseract.image_to_string(image, config=custom_config).strip()
    print(f"Recognized as speaker: {res}")
    return res


def add_speaker_to_stat(speaker_name):
    if speaker_name:
        if speaker_name in stats:
            stats[speaker_name] += 1
        else:
            stats[speaker_name] = 1


def process(processed_image):
    active_speaker_contour = find_speaker_contour(processed_image)
    active_speaker_cropped = crop_by_contour(active_speaker_contour, processed_image)
    # show(active_speaker_cropped, "Active speaker")
    prepared_ocr_image = prepare_for_ocr(active_speaker_cropped)
    show(prepared_ocr_image, "Prepared for OCR")
    speaker_name = get_speaker_name(prepared_ocr_image)
    add_speaker_to_stat(speaker_name)


def loop(scheduler):
    captured_image = get_next_image()
    process(captured_image)
    get_stats()
    scheduler.enter(1, 1, loop, (scheduler,))


def start_loop():
    s = sched.scheduler(time.time, time.sleep)

    s.enter(1, 1, loop, (s,))
    s.run()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    start_loop()

# $ sudo apt install tesseract-ocr
