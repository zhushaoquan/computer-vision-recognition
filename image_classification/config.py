class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097  # visdom 端口
    load_model_dir = "./checkpoints/epoch_50.pth.tar"
    resum_model_dir = None
    save_model_dir = "./checkpoints"
    model = 'resnet50'
    result_file = 'result.csv'
    train_data_dir = '/mnt/HD_2TB/hua/github/learning_torch/cnn/data/256_ObjectionCategoties'
    batch_size = 16
    num_workers = 4
    lr = 0.001
    weight_decay = 1e-4
    max_epoch = 10000
    label_index_dict = {
        'bear': 0,
        'chimp': 1,
        'giraffe': 2,
        'gorilla': 3,
        'llama': 4,
        'ostrich': 5,
        'porcupine': 6,
        'skunk': 7,
        'triceratops': 8,
        'zebra': 9
    }


opt = DefaultConfig()
