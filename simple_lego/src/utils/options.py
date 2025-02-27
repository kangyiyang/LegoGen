import argparse
import os
import ast

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
    self.parser.add_argument('--dataset_root', type=str, default=f'{dataset_path}/preprocessed_dataset', help='the path building dataset')
    self.parser.add_argument('--new_bricks_path', type=str, default=f'{dataset_path}/bricks/new_bricks', help='the path of new_bricks')
    self.parser.add_argument('--original_bricks_path', type=str, default=f'{dataset_path}/bricks/original_bricks', help='the path of original_bricks')
    self.parser.add_argument('--data_size', type=int, default=30, help='the size of data for every class')
    # self.parser.add_argument('--models_path', type=str, default=f'{dataset_path}/conditional_generation dataset/models', help='the path of conditional models')
    # self.parser.add_argument('--images_path', type=str, default=f'{dataset_path}/conditional_generation dataset/images', help='the rendering images of conditional models')
    
    self.parser.add_argument('--conditional_dataset_root', type=str, default=f'{dataset_path}/conditional_generation dataset', help='the path of conditional dataset')
    self.parser.add_argument('--patch_h', type=int, default=75, help='image patch of h for dino')
    self.parser.add_argument('--patch_w', type=int, default=50, help='image patch of w for dino')


    # model
    self.parser.add_argument('--ckpt_dir', type=str, default='lego/related_work/llama/llama-2-13b', help='load pretrained-llama model')
    # self.parser.add_argument('--model_params_path', type=str, default='lego/related_work/llama/llama-2-13b', help='model params path')
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
    self.parser.add_argument('--text_dim', type=int, default=128, help='the dim of input_image for the image_cross_attention')
    # self.parser.add_argument('--image_model_dir', type=str, default='lego/src/model/dinov2', help='load pretrained-dinov2 model')
    self.parser.add_argument('--text_model_size', type=str, default='dinov2_vits14', help='dinov2 size')


    # train
    self.parser.add_argument('--epochs', type=int, default=45, help='Number of epochs to train.')
    self.parser.add_argument('--batch_size', type=int, default=1)
    self.parser.add_argument('--print_freq', type=int, default=100, help='freq for loss print')
    self.parser.add_argument('--save_freq', type=int, default=100, help='freq for save checkpoint')
    self.parser.add_argument('--val_freq', type=int, default=5, help='freq for val')
    self.parser.add_argument('--save_by_iter', action='store_true', help='use iters to name checkpoint')
    self.parser.add_argument('--save_model_dir', type=str, default='/NASdata/yyk',help='the dir to save model')
    self.parser.add_argument('--train_ratio', type=float, default=0.9)
    self.parser.add_argument('--accuracy', action='store_true', help='assessment criteria')
    self.parser.add_argument('--max_norm', type=float, default=1)

    self.parser.add_argument('--mix_train', action='store_true', help='use mix train or not')
    self.parser.add_argument('--max_mix', type=float, default=0.5, help='the max ratio for mix_train')
    # self.parser.add_argument('--pretrain_model_path', type=str, default='val', help='load pretrained-bricks model')
    
    # loss
    self.parser.add_argument('--niu', type=float, default=0.1, help='weight for brick loss')
    self.parser.add_argument('--lamda', type=float, default=1, help='weight for trans loss')
    self.parser.add_argument('--miu', type=float, default=0.1, help='weight for rot loss')
    self.parser.add_argument('--focal_alpha', type=float, default=0.25, help='focal loss param')
    self.parser.add_argument('--focal_gamma', type=float, default=2, help='focal loss param')

    # metric
    self.parser.add_argument('--metric', type=str, default='acc', help='best metric to save. help:[loss,acc]')
    self.parser.add_argument('--alpha', type=float, default=0.5, help='weight for brick acc metric')
    self.parser.add_argument('--gamma', type=float, default=0.5, help='weight for position iou metric')
    # self.parser.add_argument('--delta', type=float, default=0.2, help='weight for rot similarity metric')
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
    self.parser.add_argument('--test_input_dir', type=str, default='./demo/demo_input', help='path to test demo result')
    self.parser.add_argument('--demo_input_dir', type=str, default='./demo/demo_input', help='path for demo input')
    self.parser.add_argument('--gt_demo', action='store_true', help='use gt to demo')
    self.parser.add_argument('--demo_acc', action='store_true', help='caculate demo acc to print')
    self.parser.add_argument('--save_demo_acc', action='store_true', help='save demo acc or not')
    
    # self.parser.add_argument('--pretrain_graph_path', type=str, default=f'{os.getcwd()}/GenerativeLEGO/pretrained_GIN.h5', help='the pretrain graph path')

    # render
    self.parser.add_argument('--brick_obj_path', type=str, default=f'{dataset_path}/brick_obj', help='the path of all brick objs')
    self.parser.add_argument('--color_config_path', type=str, default=f'{dataset_path}/complete/LDConfig.ldr', help='color config file')
    self.parser.add_argument('--elev', type=float, default=30, help='elevation angle for rendering')
    self.parser.add_argument('--azim', type=float, default=315, help='azimuth angle for rendering')
    self.parser.add_argument('--save_obj', action='store_true', help='save obj or not')
    

    # baseline new
    self.parser.add_argument('--baseline_model', type=str, default='epoch_0130.h5', help='load retrained-baseline model')
    self.parser.add_argument('--baseline', action='store_true', help='caculate baseline acc')
    self.parser.add_argument('--baseline_render', action='store_true', help='render baseline ldrs')

    # baseline ori
    self.parser.add_argument('--auto_implied_edges', default = 'False', choices = ['True', 'False'], help='Whether we want to manually add all implied edges for the model')
    self.parser.add_argument('--max_generated_graph_size', default = 235, type = int, help='Max generated graph size')
    self.parser.add_argument('--max_edges_per_node', default = 12, type = int, choices = [1000, 12], 
                            help='Max edges per node. 1000 is basically unlimited, 12 is the maximum number of edges \
                            we can add to a 4x2 LEGO brick and still have a valid graph')
    self.parser.add_argument('--edge_generation', type = str, choices = ['ordinal', 'softmax'], default = 'softmax', 
                            help = 'method for generating edges')
    self.parser.add_argument('--edge_embedding', type = str, default = 'embedding', choices = ['one-hot', 'embedding', 'ordinal'], 
                            help = 'method for embedding/encoding edge types in the graph')
    self.parser.add_argument('--class_conditioning', default = 'embedding', choices = ['None', 'one-hot', 'embedding'],
                            help = 'Which type of class-conditioning to use', type = str)
    self.parser.add_argument('--class_conditioning_size', default = 25, type = int,
                            help = 'The size of the class condition embedding (if class-condition == embedding)')
    self.parser.add_argument('--lr_decay_rate', default = 0.85, type = float, help = 'The learning rate decay rate')
    self.parser.add_argument('--lr_step_size', default = 50, type = int, help = 'How often to decay the lr by lr_decay_rate')
    self.parser.add_argument('--node_hidden_size', default = 80, type = int, help = 'The hidden dimensionality of each node')
    self.parser.add_argument('--num_prop_rounds', default = 2, type = int, help = 'The number of graph propagation rounds to do')
    self.parser.add_argument('--edge_hidden_size', default = 80, type = int,  help = 'The edge hidden size to use')
    self.parser.add_argument('--num_decision_layers', default = 4, type = int, help = 'The number of layers to use in the decision modules')
    self.parser.add_argument('--decision_layer_hidden_size', default = 40, type = int, help = 'The number of neurons in each decision hidden layer')
    self.parser.add_argument('--num_propagation_mlp_layers', default = 4, type = int,help = 'The number of layers in each MLP in graph prop')
    self.parser.add_argument('--prop_mlp_hidden_size', default = 40, type = int,  help = 'The hidden size of the mlp in graph prop')
    self.parser.add_argument('--include_augmented', default = 'True', choices = ['True', 'False'], 
                             help = 'Whether to include all dataset augmentations (90, 180, 270 deg rotations)')
    self.parser.add_argument('--missing_implied_edges_isnt_error', default = 'True', choices = ['True', 'False'], 
                            help = 'Whether a model not adding implied edges counts as an error or not')
    self.parser.add_argument('--valid_split', default = 0.15, type = float, help='The proportion of the dataset to be used for validation')
    self.parser.add_argument('--force_valid', default = 'False', choices = ['False'])
    self.parser.add_argument('--stop_generation', default = 'all_errors', choices = ['all_errors', 'one_error', 'None'],
                            help='When to stop generating a graph. Can stop once a graph contains all errors, a single or error, or when the model \
                            decides to stop regardless of any errors')


  def return_args(self):
    args = self.parser.parse_args()

    args.auto_implied_edges = ast.literal_eval(args.auto_implied_edges)
    args.missing_implied_edges_isnt_error = ast.literal_eval(args.missing_implied_edges_isnt_error)
    args.force_valid = ast.literal_eval(args.force_valid)
    args.include_augmented = ast.literal_eval(args.include_augmented)
     
    if args.edge_hidden_size % 2 != 0:
        raise Exception('Edge hidden size % 2 != 0, {}'.format(args.edge_hidden_size))

    args.dataset = 'Kim-et-al'
    args.num_classes = 12

    if args.class_conditioning == 'one-hot':
        args.class_conditioning_size = args.num_classes
    
    elif args.class_conditioning == 'None':
        args.class_conditioning_size = 0

    return args
