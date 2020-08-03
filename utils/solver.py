import os
import time

import torch


class Solver(object):
    
    def __init__(self, data_loader, model, criterion, optimizer, args):
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.step = 0
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            if self.use_cuda:
                self.model.module.load_state_dict(package['state_dict'])
            else:
                self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()

            tr_avg_loss = self._run_one_epoch(epoch)

            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = tr_avg_loss  # Fake
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                model = self.model.module if self.use_cuda else self.model
                torch.save(model.serialize(model,
                                           self.optimizer, epoch + 1,
                                           tr_loss=self.tr_loss,
                                           cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)
            else:
                # Save the last model
                model = self.model.module if self.use_cuda else self.model
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(model.serialize(model,
                                           self.optimizer, epoch + 1,
                                           tr_loss=self.tr_loss,
                                           cv_loss=self.cv_loss),
                           file_path)
                print('Only save the last model %s' % file_path)

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch):
        start = time.time()
        total_loss = 0

        data_loader = self.data_loader

        for i, (data) in enumerate(data_loader):
            self.step += 1
            text_padded, input_lengths, feat_padded, stop_token_padded, encoder_mask, decoder_mask = data
            text_padded = text_padded.cuda()
            input_lengths = input_lengths.cuda()
            feat_padded = feat_padded.cuda()
            stop_token_padded = stop_token_padded.cuda()
            encoder_mask = encoder_mask.cuda()
            decoder_mask = decoder_mask.cuda()
            y_pred = self.model(text_padded, input_lengths, feat_padded, encoder_mask, decoder_mask)
            y_target = (feat_padded, stop_token_padded)
            loss = self.criterion(y_pred, y_target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.3f} | {4:.1f} ms/batch | {5} step'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1),
                          self.step),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)
