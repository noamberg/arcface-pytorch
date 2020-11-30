class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 42
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    #loss = 'focal_loss'
    loss = 'bce'
    name = backbone+'_'+metric+'_'+classify

    display = True
    finetune = False

    train_root = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/data/facebank/'
    train_list = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/data/facebank/sampeled_train.txt'

    val_list = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/data/facebank/sampeled_val.txt'
    total_dataset_list = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/data/facebank/train_val.txt'

    test_root = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/data/facebank/'
    test_list = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/data/facebank/test.txt'

    lfw_root = '/data/Datasets/lfw/lfw-align-128'
    lfw_test_list = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/lfw_test_pair.txt'

    checkpoints_path = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/checkpoints'
    load_model_path = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/models/resnet18.pth'
    test_model_path = 'C:/Users/noamb/PycharmProjects/Volcani/arcface-pytorch/checkpoints/resnet18_10.pth'

    save_interval = 10

    train_batch_size = 16  # batch size
    validation_batch_size = 4
    test_batch_size = 16

    input_shape = (1, 112, 112)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 1 # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-6
