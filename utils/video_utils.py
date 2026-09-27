import os
import cv2


def create_video_from_intermediate_results(results_path, img_format):
    """
    Creates mp4 video from saved frames,
    then deletes intermediate images,
    keeping only:
      - final image
      - out.mp4
    """

    ext = img_format[1]

    frames = sorted([
        f for f in os.listdir(results_path)
        if f.endswith(ext) and f != 'out.mp4'
    ])

    if not frames:
        print('No intermediate frames found.')
        return

    first_frame_path = os.path.join(results_path, frames[0])
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        print(f'Could not read frame: {first_frame_path}')
        return

    height, width = first_frame.shape[:2]

    out_video_path = os.path.join(results_path, 'out.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30

    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    for fname in frames:
        frame_path = os.path.join(results_path, fname)
        frame = cv2.imread(frame_path)

        if frame is not None:
            writer.write(frame)

    writer.release()

    print(f'[VIDEO SAVED] → {out_video_path}')

    # Keep ONLY the last frame
    final_frame = frames[-1]

    for fname in frames:
        if fname != final_frame:
            os.remove(os.path.join(results_path, fname))

    print(f'[FINAL IMAGE KEPT] → {final_frame}')