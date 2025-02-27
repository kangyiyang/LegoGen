import argparse
import os
import sys

class opts(object):
  def __init__(self):
      
    self.parser = argparse.ArgumentParser()

    # overall
    self.parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    self.parser.add_argument('--cuda', action='store_true', help='use cuda')
    self.parser.add_argument('--USE_TENSORBOARD', action='store_true', help='use TENSORBOARD')
    self.parser.add_argument('--result_dir', type=str, default='./result',help='result path')
    self.parser.add_argument('--pretrain', action='store_true', help='pretrain ing')
    
    # data
    dataset_path = os.path.join(os.getcwd(),'dataset')
    self.parser.add_argument('--pretrain_dataset_root', type=str, default=f'{dataset_path}/pretrain_bricks dataset/processed_models', help='the path building dataset')
    self.parser.add_argument('--new_bricks_path', type=str, default=f'{dataset_path}/bricks/new_bricks', help='the path of new_bricks')
    self.parser.add_argument('--original_bricks_path', type=str, default=f'{dataset_path}/bricks/original_bricks', help='the path of original_bricks')
    # self.parser.add_argument('--models_path', type=str, default=f'{dataset_path}/conditional_generation dataset/models', help='the path of conditional models')
    # self.parser.add_argument('--images_path', type=str, default=f'{dataset_path}/conditional_generation dataset/images', help='the rendering images of conditional models')
    
    self.parser.add_argument('--conditional_dataset_root', type=str, default=f'{dataset_path}/conditional_generation dataset', help='the path of conditional dataset')
    self.parser.add_argument('--patch_h', type=int, default=75, help='image patch of h for dino')
    self.parser.add_argument('--patch_w', type=int, default=50, help='image patch of w for dino')


    # model
    self.parser.add_argument('--ckpt_dir', type=str, default='/home/yyk/lego/related_work/llama/llama-2-13b', help='load pretrained-llama model')
    # self.parser.add_argument('--model_params_path', type=str, default='/home/yyk/lego/related_work/llama/llama-2-13b', help='model params path')
    self.parser.add_argument('--max_seq_len', type=int, default=512, help='max whole length for llama transformer')
    self.parser.add_argument('--max_gen_len', type=int, default=2047, help='max gen length for llama transformer')
    self.parser.add_argument('--temperature', type=float, default=0.6, help='Temperature value for controlling randomness in sampling')
    self.parser.add_argument('--top_p', type=float, default=0.9, help='The top-p sampling parameter for controlling diversity in generation')
    self.parser.add_argument('--max_batch_size', type=int, default=4)
    self.parser.add_argument('--position_dim', type=int, default=128, help='the dim of position embeddings')
    self.parser.add_argument('--out_pad_dim', type=int, default=1, help='the dim for padding MP')
    # self.parser.add_argument('--out_rot_dim', type=int, default=24, help='the dim for rot classification')
    
    self.parser.add_argument('--rank', type=int, default=2, help='the lora rank')
    self.parser.add_argument('--add_cross', type=int, default=4, help='frequce to add cross')
    self.parser.add_argument('--c_n_heads', type=int, default=32, help='n_heads for image cross atten')
    self.parser.add_argument('--image_dim', type=int, default=128, help='the dim of input_image for the image_cross_attention')
    self.parser.add_argument('--image_model_dir', type=str, default='/home/yyk/lego/src/model/dinov2', help='load pretrained-dinov2 model')
    self.parser.add_argument('--image_model_size', type=str, default='dinov2_vits14', help='dinov2 size')


    # train
    self.parser.add_argument('--epochs', type=int, default=45, help='Number of epochs to train.')
    self.parser.add_argument('--batch_size', type=int, default=1)
    self.parser.add_argument('--print_freq', type=int, default=100, help='freq for loss print')
    self.parser.add_argument('--save_freq', type=int, default=100, help='freq for save checkpoint')
    self.parser.add_argument('--val_freq', type=int, default=5, help='freq for val')
    self.parser.add_argument('--save_by_iter', action='store_true', help='use iters to name checkpoint')
    self.parser.add_argument('--save_model_dir', type=str, default='/NASdata/yyk',help='the dir to save model')
    self.parser.add_argument('--val_ratio', type=float, default=0.1)
    self.parser.add_argument('--accuracy', action='store_true', help='assessment criteria')
    self.parser.add_argument('--max_norm', type=float, default=1)

    self.parser.add_argument('--mix_train', action='store_true', help='use mix train or not')
    self.parser.add_argument('--max_mix', type=float, default=0.5, help='the max ratio for mix_train')
    # self.parser.add_argument('--pretrain_model_path', type=str, default='/NASdata/yyk/logs_2024-03-21-19-17/val', help='load pretrained-bricks model')
    
    # loss
    self.parser.add_argument('--niu', type=float, default=0.1, help='weight for brick loss')
    self.parser.add_argument('--lamda', type=float, default=1, help='weight for trans loss')
    self.parser.add_argument('--miu', type=float, default=0.1, help='weight for rot loss')
    self.parser.add_argument('--focal_alpha', type=float, default=0.25, help='focal loss param')
    self.parser.add_argument('--focal_gamma', type=float, default=2, help='focal loss param')

    # metric
    self.parser.add_argument('--metric', type=str, default='acc', help='best metric to save. help:[loss,acc]')
    self.parser.add_argument('--alpha', type=float, default=0.4, help='weight for brick acc metric')
    self.parser.add_argument('--gamma', type=float, default=0.4, help='weight for position iou metric')
    self.parser.add_argument('--delta', type=float, default=0.2, help='weight for rot similarity metric')
    self.parser.add_argument('--scale', type=float, default=4, help='scale for iou caculate')
    self.parser.add_argument('--threshold', type=float, default=64, help='threshold for trans mse acc')
    
    # optimization
    self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate update')
    self.parser.add_argument('--use_warm_up', action='store_true',help='use warm up or not')
    self.parser.add_argument('--warm_up_ratio', type=float, default=0.1, help='wram up steps/whole steps')
    self.parser.add_argument('--weight_decay', type=float, default=0.01, help='larger to avoid overfitting')
    self.parser.add_argument('--betas', type=tuple, default=(0.9,0.95), help='betas for AdamW')
    self.parser.add_argument('--flat_lr_ratio', type=float, default=0.8, help='not flat lr steps steps/whole steps')

    # demo
    self.parser.add_argument('--demo_result_dir', type=str, default='./demo/demo_result', help='path to save demo result')
    # self.parser.add_argument('--result_gt_demo', type=str, default='./demo/demo_result/gt_demo', help='path to save gt demo result')
    # self.parser.add_argument('--result_autoregressive_demo', type=str, default='./demo/demo_result/autoregressive_demo', help='path to save autoregressive demo result')
    self.parser.add_argument('--demo_input_dir', type=str, default='./demo/demo_input', help='path for demo input')
    # self.parser.add_argument('--demo_len', type=int, default=3, help='path for demo input')
    self.parser.add_argument('--gt_demo', action='store_true', help='use gt to demo')
    self.parser.add_argument('--demo_acc', action='store_true', help='caculate demo acc to print')
    self.parser.add_argument('--save_demo_acc', action='store_true', help='save demo acc or not')

    # render
    self.parser.add_argument('--brick_objs_path', type=str, default=f'{dataset_path}/bricks/brick_objs', help='the path of all brick objs')
    self.parser.add_argument('--color_config_path', type=str, default=f'{dataset_path}/complete/LDConfig.ldr', help='color config file')
    self.parser.add_argument('--elev', type=float, default=30, help='elevation angle for rendering')
    self.parser.add_argument('--azim', type=float, default=315, help='azimuth angle for rendering')
    self.parser.add_argument('--save_obj', action='store_true', help='save obj or not')


  def return_args(self):
    return self.parser.parse_args()
