import os
import shutil


def copy_weights(arg, epoch):
    model_name = arg.arch
    epochs = arg.epochs
    batch_size = arg.batch_size
    lr = arg.lr
    ds = arg.data_format.strip()
    fpn = arg.routing_name_list if arg.routing_name_list is not None else ''
    backbone = arg.backbone if arg.backbone is not None else ''

    # folder_name = '%s_epoch%d_bs%d_lr%.1e_%s' % \
    #               (model_name, epochs, batch_size, learning_rate, datasets)
    folder_name = f"{model_name}_epoch{epochs}_bs{batch_size}_lr{lr}_{ds}{list_to_str(fpn)}{backbone}"

    # print(folder_name)
    folder_path = os.path.join('./data/weights', folder_name)
    # print('making dir ', folder_path)
    os.makedirs(folder_path, exist_ok=True)

    new_checkpoint = f'checkpoint_epoch{epoch+1}.pth.tar'
    origin_checkpoint = 'checkpoint.pth.tar'
    model_best_name = 'model_best.pth.tar'

    print("copy file from %s to %s" % (
        os.path.join('./data', origin_checkpoint),
        os.path.join(folder_path, new_checkpoint)))
    shutil.copyfile(os.path.join('./data', origin_checkpoint),
                    os.path.join(folder_path, new_checkpoint))

    print("copy file from %s to %s" % (
        os.path.join('./data', model_best_name),
        os.path.join(folder_path, model_best_name)))
    shutil.copyfile(os.path.join('./data', model_best_name),
                    os.path.join(folder_path, model_best_name))


def list_to_str(str_list: list):
    out = ""
    if str_list is None:
        return out
    for item in str_list:
        if item is None:
            return out
        out += "_" + item

    return str(out)
