import zmq
import cv2
import modules.globals
import numpy as np
import threading
import time
import io
import modules.processors.frame.core
from tqdm import tqdm
from modules.typing import Face, Frame
from typing import Any,List
from modules.core import update_status
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
import zlib
import subprocess
from cv2 import VideoCapture
import queue
NAME = 'DLC.REMOTE-PROCESSOR'

context = zmq.Context()
SOCKET_TIMEOUT_MS = 15_000
RESULT_TIMEOUT_MS = 120_000
CHUNK_SIZE = 4 * 1024 * 1024

# Socket to send messages on
def push_socket(address) -> zmq.Socket:
    sender_sock = context.socket(zmq.REQ)
    sender_sock.setsockopt(zmq.LINGER, 0)
    sender_sock.setsockopt(zmq.RCVTIMEO, SOCKET_TIMEOUT_MS)
    sender_sock.setsockopt(zmq.SNDTIMEO, SOCKET_TIMEOUT_MS)
    sender_sock.connect(address)
    return sender_sock
def pull_socket(address, timeout_ms: int = SOCKET_TIMEOUT_MS) -> zmq.Socket:
    sender_sock = context.socket(zmq.REP)
    sender_sock.setsockopt(zmq.LINGER, 0)
    sender_sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sender_sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sender_sock.connect(address)
    return sender_sock

def pre_check() -> bool:
    if not modules.globals.push_addr and not modules.globals.pull_addr:
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True

def stream_frame(temp_frame: Frame,stream_out: subprocess.Popen[bytes],stream_in: subprocess.Popen[bytes]) -> Frame:
    temp_framex = swap_face_remote(temp_frame,stream_out,stream_in)

    return temp_framex

def process_frame(source_frame: Frame, temp_frame: Frame)-> Frame:
    temp_framex = swap_frame_face_remote(source_frame,temp_frame)

    return temp_framex
def send_data(sender: zmq.Socket, face_bytes: bytes, metadata: dict, address: str) -> None:
    chunk_size = CHUNK_SIZE
    total_chunk = (len(face_bytes) + chunk_size - 1) // chunk_size #len(face_bytes) // chunk_size + 1
    new_metadata = {'total_chunk': total_chunk}
    metadata.update(new_metadata)
    # Send metadata first
    sender.send_json(metadata)
    # Wait for acknowledgment for metadata
    ack = sender.recv_string()
    with tqdm(total=total_chunk, desc="Sending chunks", unit="chunk") as pbar:
        for i in range(total_chunk):
            chunk = face_bytes[i * chunk_size:(i + 1) * chunk_size]
            # Send the chunk
            sender.send(chunk)
            # Wait for acknowledgment after sending each chunk
            ack = sender.recv_string()
            pbar.set_postfix_str(f'Chunk {i + 1}/{total_chunk} ack: {ack}')
            pbar.update(1)
       
    # Send a final message to indicate all chunks are sent
    sender.send(b"END")
    # Wait for the final reply
    final_reply_message = sender.recv_string()
    print(f"Received final reply: {final_reply_message}")
#client.py
def send_source_frame(source_face: Frame)-> None:
    sender = push_socket(modules.globals.push_addr)
    try:
        source_face_bytes = source_face.tobytes()
        metadata = {
            'manyface':(modules.globals.many_faces),
            'enhance': bool(modules.globals.remote_face_enhancer),
            'dtype_source':str(source_face.dtype),
            'shape_source':source_face.shape,
            'size':'640x480',
            'fps':'60'
            #'shape_temp':temp_frame.shape
        }
        send_data(sender, source_face_bytes, metadata,modules.globals.push_addr)
    finally:
        sender.close(0)

def send_temp_frame(temp_face: Frame)-> None:
    sender = push_socket(modules.globals.push_addr_two)
    try:
        source_face_bytes = temp_face.tobytes()
        metadata = {
            'manyface':(modules.globals.many_faces),
            'dtype_temp':str(temp_face.dtype),
            'shape_temp':temp_face.shape,
            
            #'shape_temp':temp_frame.shape
        }
        send_data(sender, source_face_bytes, metadata,modules.globals.push_addr_two)
    finally:
        sender.close(0)

def receive_processed_frame() -> Frame:
    pull_socket_ = pull_socket(modules.globals.pull_addr, RESULT_TIMEOUT_MS)
    try:
        try:
            meta_data_json = pull_socket_.recv_json()
        except zmq.Again as exception:
            raise TimeoutError(
                f'Remote processor returned no result within {RESULT_TIMEOUT_MS // 1000} seconds. '
                'Check the Colab server output for an inference or face-detection error.'
            ) from exception
        print(meta_data_json)
        if meta_data_json.get('status') == 'error':
            pull_socket_.send_string("ACK")
            raise RuntimeError(f"Remote processing failed: {meta_data_json.get('error', 'unknown error')}")
        total_chunk = meta_data_json['total_chunk']
        #num_data = meta_data_json['datasize']
        # Send acknowledgment for metadata
        pull_socket_.send_string("ACK")
        # Receive the array bytes
        source_array_bytes =b'' 
        with tqdm(total=total_chunk, desc="Receiving chunks", unit="chunk") as pbar:
            for i in range(total_chunk):
                chunk = pull_socket_.recv()
                source_array_bytes += chunk
                pull_socket_.send_string(f"ACK {i + 1}/{total_chunk}")
                pbar.set_postfix_str(f'Chunk {i + 1}/{total_chunk}')
                pbar.update(1)
            

        end_message = pull_socket_.recv()
        if end_message == b"END":
            pull_socket_.send_string("Final ACK")
        
        # Deserialize the bytes back to an ndarray
        source_array = np.frombuffer(source_array_bytes, dtype=np.dtype(meta_data_json['dtype_source'])).reshape(meta_data_json['shape_source'])

        return source_array.copy()
    finally:
        pull_socket_.close(0)
def send_streams(cap: VideoCapture) -> subprocess.Popen[bytes]:
    
    ffmpeg_command = [
        'ffmpeg',
        '-fflags', 'nobuffer',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s',  f"{modules.globals.live_width}x{modules.globals.live_height}",
        '-r',  str(modules.globals.live_fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        
        #'-flags', 'low_delay',
        #'-rtbufsize', '100M',
        '-g', modules.globals.gof,#'15',  # GOP size set to 1 to make each frame a keyframe
        '-b:v', modules.globals.bitrate,#'800k',  # Set bitrate to 1 Mbps
        '-maxrate', modules.globals.maxrate,#'800k',  
        '-bufsize', modules.globals.bufsize,#'800k',
        '-f', 'mpegts', modules.globals.push_addr_two #'tcp://127.0.0.1:5552'
    ]


    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    return ffmpeg_process
def recieve_streams(cap: VideoCapture)->subprocess.Popen[bytes]:
    ffmpeg_command_recie = [
    'ffmpeg',
    '-i',modules.globals.pull_addr, #'tcp://127.0.0.1:5553',
    '-f','rawvideo',
    '-pix_fmt','bgr24',
    '-s', f'{modules.globals.live_width}x{modules.globals.live_height}',
    'pipe:1'
    ]
    
    ffmpeg_process_com = subprocess.Popen(ffmpeg_command_recie, stdout=subprocess.PIPE)
    return ffmpeg_process_com

def write_to_stdin(temp_frame: Frame,queue: queue.Queue, stream_out: subprocess.Popen):
   
    #temp_frame = queue.get()
    temp_frame_bytes = temp_frame.tobytes()
    stream_out.stdin.write(temp_frame_bytes)
def read_from_stdout(stream_in: subprocess.Popen, output_queue: queue.Queue):
    
    raw_frame = stream_in.stdout.read(960 * 540 * 3)
    
    
    if len(raw_frame) != modules.globals.live_width * modules.globals.live_height * 3:
        raise EOFError('Remote live stream ended before a complete frame was received')
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((modules.globals.live_height, modules.globals.live_width, 3))
    output_queue.put(frame)  


def run_live_virtual_camera(source_path: str, stop_event: threading.Event = None) -> None:
    """Bridge a physical camera through Colab into a Windows virtual camera."""
    if not source_path:
        raise ValueError('Select a source face image before starting live mode')
    try:
        import pyvirtualcam
        from pyvirtualcam import PixelFormat
    except ImportError as exception:
        raise RuntimeError('Install pyvirtualcam in the project venv') from exception

    stop_event = stop_event or threading.Event()
    cap = cv2.VideoCapture(modules.globals.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, modules.globals.live_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, modules.globals.live_height)
    cap.set(cv2.CAP_PROP_FPS, modules.globals.live_fps)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open webcam index {modules.globals.camera_index}')

    source = cv2.imread(source_path)
    if source is None:
        cap.release()
        raise ValueError(f'Unable to read source face image: {source_path}')

    stream_out = stream_in = None
    try:
        send_source_frame(source)
        stream_out = send_streams(cap)
        stream_in = recieve_streams(cap)
        device = modules.globals.virtual_camera
        camera_options = {'device': device} if device else {}
        with pyvirtualcam.Camera(
            width=modules.globals.live_width,
            height=modules.globals.live_height,
            fps=modules.globals.live_fps,
            fmt=PixelFormat.BGR,
            print_fps=True,
            **camera_options,
        ) as virtual_cam:
            print(f'Virtual camera ready: {virtual_cam.device} ({virtual_cam.backend})')
            print('Use that camera in Zoom/Discord/Teams/OBS. Press Ctrl+C to stop.')
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Physical webcam stopped producing frames')
                if frame.shape[1] != modules.globals.live_width or frame.shape[0] != modules.globals.live_height:
                    frame = cv2.resize(frame, (modules.globals.live_width, modules.globals.live_height))
                stream_out.stdin.write(np.ascontiguousarray(frame).tobytes())
                raw = stream_in.stdout.read(modules.globals.live_width * modules.globals.live_height * 3)
                if len(raw) != modules.globals.live_width * modules.globals.live_height * 3:
                    raise EOFError('Colab live output stream closed')
                processed = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (modules.globals.live_height, modules.globals.live_width, 3)
                )
                virtual_cam.send(processed)
                virtual_cam.sleep_until_next_frame()
    finally:
        cap.release()
        for process in (stream_out, stream_in):
            if process is not None and process.poll() is None:
                process.terminate()
def swap_face_remote(temp_frame: Frame,stream_out:subprocess.Popen[bytes],stream_in: subprocess.Popen[bytes]) -> Frame:
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # Start threads for stdin and stdout
    write_thread = threading.Thread(target=write_to_stdin, args=(input_queue, stream_out))
    read_thread = threading.Thread(target=read_from_stdout, args=(input_queue, stream_in, output_queue))

    read_thread.start()
    write_thread.start()
    
    
    # Send the frame to the stdin thread
    input_queue.put(temp_frame)
    #processed_frame = []
    # Wait for the processed frame from the stdout thread
    processed_frame = output_queue.get()

    # Stop the threads
    input_queue.put(None)
    write_thread.join()
    read_thread.join()

    return processed_frame
    
    
def swap_frame_face_remote(source_frame: Frame,temp_frame: Frame) -> Frame:
    # Run request/response work on the caller thread so transport failures are
    # raised instead of leaving the GUI blocked on an empty queue forever.
    send_source_frame(source_frame)
    send_temp_frame(temp_frame)
    return receive_processed_frame()


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    source_frame = cv2.imread(source_path)
    frames = []
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        frames.append(temp_frame)
        ''' 
        try:
            result = process_frame(source_frame, temp_frame)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            print(exception)
            pass'''
        if progress:
            progress.update(1)
    # Convert list of frames into a 4D NumPy array
    temp_frame = np.stack(frames, axis=0)
    print("Video Dimentsion",temp_frame.ndim,temp_frame.shape)
    result = process_frame(source_frame, temp_frame)
    print("Video Dimension result",result.shape)
    for i,frame in enumerate(result):
        print(f"write to {temp_frame_paths[i]}" )
        cv2.imwrite(temp_frame_paths[i], frame)

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_frame = cv2.imread(source_path)
    target_frame = cv2.imread(target_path)
    result = process_frame(source_frame, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    #modules.processors.frame.core.remote_process_video(source_path, temp_frame_paths, process_frames)
    process_frames(source_path, temp_frame_paths)
