import pyvirtualcam

print('pyvirtualcam version:', pyvirtualcam.__version__)

print('\nTrying to list cameras...')
try:
    cameras = pyvirtualcam.Camera.list_cameras()
    if cameras:
        print('Available cameras:')
        for cam in cameras:
            print(f"  - '{cam}'")
    else:
        print('  (none found)')
except Exception as e:
    print(f'Error listing cameras: {e}')

print('\nTrying to create camera with default settings...')
try:
    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        print(f'Successfully created camera: {cam.device} (backend: {cam.backend})')
except Exception as e:
    print(f'Error creating default camera: {e}')

print('\nTrying to create camera with OBS Virtual Camera...')
try:
    with pyvirtualcam.Camera(width=640, height=480, fps=30, device='OBS Virtual Camera') as cam:
        print(f'Successfully created camera: {cam.device} (backend: {cam.backend})')
except Exception as e:
    print(f'Error creating OBS Virtual Camera: {e}')
