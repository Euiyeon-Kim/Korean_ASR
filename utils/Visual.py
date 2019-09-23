'''
    Have to open visdom server first
    'python -m visdom.server' or 'visdom'
'''
import sys
import numpy as np
from visdom import Visdom

class Visual():
    def __init__(self, num_batches):
        self.viz = Visdom()
        self.num_batches = num_batches
        self.cur_epoch = 1
        self.cur_batch = 1
        self.terms = {}                                # Running average로 매 배치마다 terminal에 logging
        self.losses = {}                               # Running average로 매 epoch마다 visualize할 data
        self.visualize_losses = {}                     # Windows for visualized losses

    def log(self, losses=None):
        '''
            losses: visdom에 visualize할 losses dictionary
        '''
        # Running average for visualized loss logging
        for loss_name, loss in losses.items():
            if loss_name not in self.losses:
                self.losses[loss_name] = loss
            else:
                self.losses[loss_name] += loss

        # Visualize logging for losses - every epoch
        if self.cur_batch % self.num_batches == 0:
            for loss_name, loss in self.losses.items():
                if loss_name not in self.visualize_losses:  # 새로운 loss일 경우 윈도우 생성
                    self.visualize_losses[loss_name] = self.viz.line(X=np.array([self.cur_epoch]), Y=np.array([loss/self.num_batches]),
                                                                     opts={'title':loss_name, 'xlabel':'epochs', 'ylabel':loss_name})
                else:                                       # 이전 윈도우에 info append
                    self.viz.line(X=np.array([self.cur_epoch]), Y=np.array([loss/self.num_batches]), win=self.visualize_losses[loss_name], update='append')
                self.losses[loss_name] = 0.0

            self.cur_epoch += 1
            self.cur_batch = 1
        else:
            self.cur_batch += 1
