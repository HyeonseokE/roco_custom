import sys
sys.path.append('/home/hscho/.local/lib/python3.8/site-packages')

import os
import ffmpeg

def merge_videos_in_directory(run_folder_path: str, output_path: str):

    output_file_name = "merged_video.mp4"
    output_path = os.path.join(output_path, output_file_name)

    videos = []
    for folder in os.listdir(run_folder_path):
        if len(folder.split(".")) == 1:  # 확장자 없는 폴더만
            folder_path = os.path.join(run_folder_path, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".mp4"):
                    file_path = os.path.join(folder_path, file)
                    videos.append(file_path)

    # 비디오 sorting
    sorted_videos = sorted(videos)

    # 임시 concat 리스트 파일 생성
    list_file = 'file_list.txt'
    with open(list_file, 'w') as f:
        for video_file in sorted_videos:
            f.write(f"file '{os.path.abspath(video_file)}'\n")

    # ffmpeg concat demuxer로 병합 (재인코딩 없이 빠름)
    try:
        (
            ffmpeg
            .input(list_file, format='concat', safe=0)
            .output(output_path, c='copy')
            .run(overwrite_output=True)
        )
        print(f"병합 완료: {output_path}")
    except ffmpeg.Error as e:
        print('ffmpeg error:', e.stderr.decode())
    finally:
        # 임시 파일 삭제
        if os.path.exists(list_file):
            os.remove(list_file)


if __name__ == "__main__":
    run_folder_path = "/home/hscho/hscho/docker/src/robot-collab/data/test/run_2"
    output_path = "./"
    merge_videos_in_directory(run_folder_path, output_path)
