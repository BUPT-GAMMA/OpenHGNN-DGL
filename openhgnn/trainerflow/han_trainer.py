import dgl
import torch
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping
from ..sampler import HANSampler


@register_flow("han_trainer")
class HAN(BaseFlow):
    r"""
    HAN flow.
    """

    def __init__(self, args):
        """

        Attributes
        ------------
        category: str
            The target node type to predict
        num_classes: int
            The number of classes for category node type

        """

        super(HAN, self).__init__(args)
        self.args.category = self.task.dataset.category
        self.category = self.args.category

        self.num_classes = self.task.dataset.num_classes

        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            self.logger.info('[NC Specific] Modify the out_dim with num_classes')
            args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()

        if self.args.prediction_flag:
            self.pred_idx = self.task.dataset.pred_idx

        self.labels = self.task.get_labels().to(self.device)

        if self.args.mini_batch_flag:

            sampler = HANSampler(g=self.hg, category=self.category, meta_paths_dict=self.args.meta_paths_dict,
                                 num_neighbors=20)
            self.train_loader = dgl.dataloading.DataLoader(
                self.hg.cpu(), {self.category: self.train_idx.cpu()}, sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True, num_workers=0)
            self.val_loader = dgl.dataloading.DataLoader(
                self.hg.to('cpu'), {self.category: self.valid_idx.to('cpu')}, sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True, num_workers=0)
            if self.args.test_flag:
                self.test_loader = dgl.dataloading.DataLoader(
                    self.hg.to('cpu'), {self.category: self.test_idx.to('cpu')}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True, num_workers=0)
            if self.args.prediction_flag:
                self.pred_loader = dgl.dataloading.DataLoader(
                    self.hg.to('cpu'), {self.category: self.pred_idx.to('cpu')}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)

    def preprocess(self):

        super(HAN, self).preprocess()

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            else:
                train_loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                modes = ['train', 'valid']
                if self.args.test_flag:
                    modes = modes + ['test']
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, losses = self._mini_test_step(modes=modes)
                    # train_score, train_loss = self._mini_test_step(modes='train')
                    # val_score, val_loss = self._mini_test_step(modes='valid')
                else:
                    metric_dict, losses = self._full_test_step(modes=modes)
                val_loss = losses['valid']
                self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
                                       + self.logger.metric2str(metric_dict))
                early_stop = stopper.loss_step(val_loss, self.model)
                if early_stop:
                    self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                    break

        stopper.load_model(self.model)
        if self.args.prediction_flag:
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                indices, y_predicts = self._mini_prediction_step()
            else:
                y_predicts = self._full_prediction_step()
                indices = torch.arange(self.hg.num_nodes(self.category))
            return indices, y_predicts

        if self.args.test_flag:
            if self.args.dataset[:4] == 'HGBn':
                # save results for HGBn
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, val_loss = self._mini_test_step(modes=['valid'])
                else:
                    metric_dict, val_loss = self._full_test_step(modes=['valid'])
                self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
                self.model.eval()
                with torch.no_grad():
                    h_dict = self.model.input_feature()
                    logits = self.model(self.hg, h_dict)[self.category]
                    self.task.dataset.save_results(logits=logits, file_path=self.args.HGB_results_path)
                return dict(metric=metric_dict, epoch=epoch)
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                metric_dict, _ = self._mini_test_step(modes=['valid', 'test'])
            else:
                metric_dict, _ = self._full_test_step(modes=['valid', 'test'])
            self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
            return dict(metric=metric_dict, epoch=epoch)

    def _full_train_step(self):
        self.model.train()
        h_dict = self.model.input_feature()
        self.hg = self.hg.to(self.device)
        h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
        logits = self.model(self.hg, h_dict)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self, ):
        self.model.train()
        loss_all = 0.0
        loader_tqdm = tqdm(self.train_loader, ncols=120)
        for i, (input_nodes_dict, seeds, block_dict) in enumerate(loader_tqdm):
            block_dict = {meta_path_name: blk.to(self.device) for meta_path_name, blk in block_dict.items()}
            seeds = seeds[self.category]  # out_nodes, we only predict the nodes with type "category"
            emb_dict = {}
            for meta_path_name, input_nodes in input_nodes_dict.items():
                emb_dict[meta_path_name] = self.model.input_feature.forward_nodes(input_nodes)

            lbl = self.labels[seeds].to(self.device)
            logits = self.model(block_dict, emb_dict)[self.category]
            loss = self.loss_fn(logits, lbl)
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / (i + 1)

    def _full_test_step(self, modes, logits=None):
        """
        Parameters
        ----------
        mode: list[str]
            `train`, 'test', 'valid' are optional in list.
        logits: dict[str, th.Tensor]
            given logits, default `None`.

        Returns
        -------
        metric_dict: dict[str, float]
            score of evaluation metric
        info: dict[str, str]
            evaluation information
        loss: dict[str, float]
            the loss item
        """
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = logits if logits else self.model(self.hg, h_dict)[self.category]
            masks = {}
            for mode in modes:
                if mode == "train":
                    masks[mode] = self.train_idx
                elif mode == "valid":
                    masks[mode] = self.valid_idx
                elif mode == "test":
                    masks[mode] = self.test_idx

            metric_dict = {key: self.task.evaluate(logits, mode=key) for key in masks}
            loss_dict = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
            return metric_dict, loss_dict

    def _mini_test_step(self, modes):
        self.model.eval()
        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'train':
                    loader_tqdm = tqdm(self.train_loader, ncols=120)
                elif mode == 'valid':
                    loader_tqdm = tqdm(self.val_loader, ncols=120)
                elif mode == 'test':
                    loader_tqdm = tqdm(self.test_loader, ncols=120)
                y_trues = []
                y_predicts = []
                for i, (input_nodes_dict, seeds, block_dict) in enumerate(loader_tqdm):
                    block_dict = {meta_path_name: blk.to(self.device) for meta_path_name, blk in block_dict.items()}
                    seeds = seeds[self.category]  # out_nodes, we only predict the nodes with type "category"
                    emb_dict = {}
                    for meta_path_name, input_nodes in input_nodes_dict.items():
                        emb_dict[meta_path_name] = self.model.input_feature.forward_nodes(input_nodes)
                    logits = self.model(block_dict, emb_dict)[self.category]

                    lbl = self.labels[seeds].to(self.device)
                    loss = self.loss_fn(logits, lbl)

                    loss_all += loss.item()
                    y_trues.append(lbl.detach().cpu())
                    y_predicts.append(logits.detach().cpu())
                loss_all /= (i + 1)
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)
                evaluator = self.task.get_evaluator(name='f1')
                metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                loss_dict[mode] = loss
        return metric_dict, loss_dict

    def _full_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = self.model(self.hg, h_dict)[self.category]
            return logits

    def _mini_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            loader_tqdm = tqdm(self.pred_loader, ncols=120)
            indices = []
            y_predicts = []
            for i, (input_nodes_dict, seeds, block_dict) in enumerate(loader_tqdm):
                block_dict = {meta_path_name: blk.to(self.device) for meta_path_name, blk in block_dict.items()}
                seeds = seeds[self.category]
                emb_dict = {}
                for meta_path_name, input_nodes in input_nodes_dict.items():
                    emb_dict[meta_path_name] = self.model.input_feature.forward_nodes(input_nodes)
                logits = self.model(block_dict, emb_dict)[self.category]
                indices.append(seeds.detach().cpu())
                y_predicts.append(logits.detach().cpu())
            indices = torch.cat(indices, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
        return indices, y_predicts
