import argparse
import os
import random
import string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/mnt/data1_hdd/wgk/PaddleClas/tr/datasetsmuti/train')
    parser.add_argument('--save_img_list_path', type=str, default='/mnt/data1_hdd/wgk/PaddleClas/tr/datasetsmuti/train.txt')
    parser.add_argument('--save_label_map_path', type=str, default='/mnt/data1_hdd/wgk/PaddleClas/tr/datasetsmuti/label.txt')

    args = parser.parse_args()
    return args


def main(args):
    img_list = []
    label_list = []
    #img_end = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp']
    img_end=['png']
    if args.dataset_path[-1] == "/":
        args.dataset_path = args.dataset_path[:-1]
    if not os.path.exists(args.dataset_path):
        raise Exception(f"The data path {args.dataset_path} not exists.")
    else:
        label_name_list = [
            label for label in os.listdir(args.dataset_path)
            if os.path.isdir(os.path.join(args.dataset_path, label))
        ]

    for index, label_name in enumerate(label_name_list):
        for root, dirs, files in os.walk(
                os.path.join(args.dataset_path, label_name)):
            for single_file in files:
                if single_file.split('.')[-1] in img_end:
                    img_path = os.path.relpath(
                        os.path.join(root, single_file),
                        os.path.dirname(args.dataset_path))
                    img_list.append(f'{img_path} {index}')
                else:
                    print(
                        f'WARNING: File {single_file} end with {single_file.split(".")[-1]} is not supported.'
                    )
        label_list.append(f'{index} {label_name}')

    if len(img_list) == 0:
        raise Exception(f"Not found any images file in {args.dataset_path}.")

    with open(
            os.path.join(
                os.path.dirname(args.dataset_path), args.save_img_list_path),
            'w') as f:
        f.write('\n'.join(img_list))
    print(
        f'Already save {args.save_img_list_path} in {os.path.join(os.path.dirname(args.dataset_path), args.save_img_list_path)}.'
    )

    with open(
            os.path.join(
                os.path.dirname(args.dataset_path), args.save_label_map_path),
            'w') as f:
        f.write('\n'.join(label_list))
    print(
        f'Already save {args.save_label_map_path} in {os.path.join(os.path.dirname(args.dataset_path), args.save_label_map_path)}.'
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)