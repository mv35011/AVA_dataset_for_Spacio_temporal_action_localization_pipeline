def get_yolox_config():
    return {
        'config_file': 'detection/yolox_config/yolox_s.py',
        'checkpoint_file': 'detection/yolox_config/yolox_s.pth',
        'score_thresh': 0.25,
        'device': 'cuda:0'  # use 'cpu' if testing without GPU
    }



