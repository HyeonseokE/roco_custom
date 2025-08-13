import sys
sys.path.append('/home/hscho/.local/lib/python3.8/site-packages')

import os
import subprocess

def get_last_MP4_video_paths(data_folder_path: str, target_file_name):
    '''
    move final .mp4 videos to local windows
    '''
    root_dir = "test"
    root_path = os.path.join(data_folder_path, root_dir)
    run_folders = [run_folder for run_folder in sorted(os.listdir(root_path)) if "run" in run_folder]
    
    last_video_paths = []
    for run_folder in run_folders:
        run_folder_path = os.path.join(root_path, run_folder)
        for step_folder in sorted(os.listdir(run_folder_path), reverse=True):
            step_folder_path = os.path.join(run_folder_path, step_folder)
            if os.path.isdir(step_folder_path):
                if target_file_name in os.listdir(step_folder_path):
                    target_file_path = os.path.join(step_folder_path, target_file_name)
                    last_video_paths.append(os.path.join(step_folder_path, change_file_name_path(target_file_path)))
                    break

    return last_video_paths

def send_files_via_scp(video_paths, remote_user, remote_ip, remote_path):
    for file_path in video_paths:
        try:
            # scp 명령어 구성
            scp_command = [
                "scp",
                file_path,
                f"{remote_user}@{remote_ip}:{remote_path}"
            ]
            # 명령어 실행
            result = subprocess.run(scp_command, check=True, capture_output=True, text=True)
            print(f"[성공] {file_path} 전송 완료")
        except subprocess.CalledProcessError as e:
            print(f"[실패] {file_path} 전송 실패")
            print("오류 메시지:", e.stderr)

def change_file_name_path(target_file_path):

    run, step = target_file_path.split('/')[-3:-1]
    changed_file_name = f"execute({run}_{step}).mp4"
    changed_target_file_path = os.path.join("/".join(target_file_path.split('/')[:-1]), changed_file_name)

    if not os.path.exists(changed_target_file_path):
        try:
            # 명령어 구성
            rename_command = [
                "mv",
                target_file_path,
                changed_target_file_path
            ]
            # 명령어 실행
            result = subprocess.run(rename_command, check=True, capture_output=True, text=True)
            print(f"[성공] {target_file_path} -> {changed_target_file_path}")
        except subprocess.CalledProcessError as e:
            print(f"[실패] {target_file_path} -> {changed_target_file_path}")
            print("오류 메시지:", e.stderr)
    return changed_target_file_path

if __name__ == "__main__":
    data_folder_path = "/home/hscho/hscho/docker/src/robot-collab/data"
    target_file_name = "execute.mp4"
    windows_user_name = "hscho"
    # windows_ip = 
    # remote_path = 
    # output_path = "./"
    last_video_paths = get_last_MP4_video_paths(data_folder_path = data_folder_path,
                                                target_file_name = target_file_name)
    # send_files_via_scp(video_paths = last_video_paths,
    #                    remote_user = windows_user_name,
    #                    remote_ip = windows_ip,
    #                    remote_path = remote_path)