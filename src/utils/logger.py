from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import if_exists

class Logger(object):
  def __init__(self, opt):
    """Create a summary writer logging to log_dir."""
    if_exists(opt.result_dir)
   
    self.time_str = time.strftime('%Y-%m-%d-%H-%M')

    args = dict((name, getattr(opt, name)) for name in dir(opt)
                if not name.startswith('_'))
    file_name = os.path.join(opt.result_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> torch version: {}\n'.format(torch.__version__))
      opt_file.write('==> cudnn version: {}\n'.format(
        torch.backends.cudnn.version()))
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')
      for k, v in sorted(args.items()):
        opt_file.write('  %s: %s\n' % (str(k), str(v)))
    
    self.log_dir = opt.result_dir + '/logs_{}'.format(self.time_str)
    save_model_path=opt.save_model_dir+'/logs_{}'.format(self.time_str)

    self.val_path=os.path.join(save_model_path,'val')
    self.train_path=os.path.join(save_model_path,'train')
    self.latest_path=os.path.join(save_model_path,'latest')
    if_exists(self.val_path)
    if_exists(self.train_path)
    if_exists(self.latest_path)
    os.system('cp {}/params.json {}/'.format(opt.ckpt_dir, self.val_path))
    os.system('cp {}/params.json {}/'.format(opt.ckpt_dir, self.train_path))
    os.system('cp {}/params.json {}/'.format(opt.ckpt_dir, self.latest_path))

    if opt.USE_TENSORBOARD:
      self.writer = SummaryWriter(log_dir=self.log_dir)
    else:
      if not os.path.exists(os.path.dirname(self.log_dir)):
        os.mkdir(os.path.dirname(self.log_dir))
      if not os.path.exists(self.log_dir):
        os.mkdir(self.log_dir)
    self.log = open(self.log_dir + '/log.txt', 'w')
    try:
      os.system('cp {}/opt.txt {}/'.format(opt.result_dir, self.log_dir))
    except:
      pass
    self.start_line = True

  def write(self, txt):
    if self.start_line:
      time_str = time.strftime('%Y-%m-%d-%H-%M')
      self.log.write('{}: {}'.format(time_str, txt))
    else:
      self.log.write(txt)  
    self.start_line = False
    if '\n' in txt:
      self.start_line = True
      self.log.flush()
  
  def close(self):
    self.log.close()
  
  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    self.writer.add_scalar(tag, value, step)