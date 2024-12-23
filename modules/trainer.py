import logging
import os
from datetime import datetime
from abc import abstractmethod
import torch
from numpy import inf

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        # Create a timestamp for the log filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Logging setup
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # File handler for logging to a file with timestamp in filename
        log_file = os.path.join(args.save_dir, f'training_{timestamp}.log')
        file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.log_file_handler = file_handler  # Store the handler for later use

        # Setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)
        # else:
        #     self._resume_checkpoint('/kaggle/working/results/iu_xray')

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                result = self._train_epoch(epoch)

                # Save logged informations into log dict
                log = {'epoch': epoch}
                log.update(result)
                self._record_best(log)

                # Print logged informations to the screen and logger
                for key, value in log.items():
                    log_message = '\t{:15s}: {}'.format(str(key), value)
                    print(log_message)  # Print to console
                    self.logger.info(log_message)  # Log to file

                # Flush the log handler after every epoch
                self._flush_log_handler()

                # Evaluate model performance according to configured metric, save best checkpoint as model_best
                best = False
                if self.mnt_mode != 'off':
                    try:
                        # Check whether model performance improved or not, according to specified metric (mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        log_message = "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric)
                        print(log_message)  # Print to console
                        self.logger.warning(log_message)  # Log to file
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        log_message = "Validation performance didn’t improve for {} epochs. Training stops.".format(self.early_stop)
                        print(log_message)  # Print to console
                        self.logger.info(log_message)  # Log to file
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, save_best=best)
        finally:
            self._finalize_logging()

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        log_message = 'Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric)
        print(log_message)  # Print to console
        self.logger.info(log_message)  # Log to file
        for key, value in self.best_recorder['val'].items():
            log_message = '\t{:15s}: {}'.format(str(key), value)
            print(log_message)  # Print to console
            self.logger.info(log_message)  # Log to file

        log_message = 'Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric)
        print(log_message)  # Print to console
        self.logger.info(log_message)  # Log to file
        for key, value in self.best_recorder['test']. items():
            log_message = '\t{:15s}: {}'.format(str(key), value)
            print(log_message)  # Print to console
            self.logger.info(log_message)  # Log to file

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            log_message = "Warning: There’s no GPU available on this machine, training will be performed on CPU."
            print(log_message)  # Print to console
            self.logger.warning(log_message)  # Log to file
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            log_message = "Warning: The number of GPU's configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
            print(log_message)  # Print to console
            self.logger.warning(log_message)  # Log to file
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        log_message = "Saving checkpoint: {} ...".format(filename)
        print(log_message)  # Print to console
        self.logger.info(log_message)  # Log to file
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            log_message = "Saving current best: model_best.pth ..."
            print(log_message)  # Print to console
            self.logger.info(log_message)  # Log to file

        # Flush the log handler whenever a checkpoint is saved
        self._flush_log_handler()

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        log_message = "Loading checkpoint: {} ...".format(resume_path)
        print(log_message)  # Print to console
        self.logger.info(log_message)  # Log to file
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        log_message = "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        print(log_message)  # Print to console
        self.logger.info(log_message)  # Log to file

    def _flush_log_handler(self):
        self.log_file_handler.flush()

    def _finalize_logging(self):
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
        self.logger.handlers.clear()





class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader,
                 val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):
        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        # Inside the _train_epoch method
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
            # print(f"Batch {batch_idx} - Images ID: {images_id}")
            # print("images shape:",images.shape)
            # Forward pass
            output = self.model(images, reports_ids, mode='train')

            # Get visual and textual features for the losses
            textual_features, visual_features, fc_feats_0, fc_feats_1 = output

            loss = self.criterion(reports_ids, reports_masks, visual_features, textual_features, images_id, fc_feats_0, fc_feats_1)

            # Backpropagation and optimization
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # Validation phase
        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                     reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        # Test phase
        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), \
                                                     reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        # Update learning rate scheduler
        self.lr_scheduler.step()

        return log

