"""SpeechBrain recipe that trains the SER task adapter (Stages 1 and 3).

The paper's SER models were trained with the EmoBox recipe
(https://github.com/emo-box/EmoBox), itself based on the SpeechBrain IEMOCAP
emotion-recognition recipe. This module is that recipe for pre-extracted
utterance-level latents: Adam on a log-softmax / NLL objective, gradient
clipping and the non-finite-loss check of ``sb.Brain``, SpeechBrain's
``ReproducibleRandomSampler`` for shuffling and a ``LinearWarmupScheduler``
that is stepped once per epoch. No checkpoint selection is done; the model
after the last epoch is used.

Requires ``speechbrain==0.5.16`` (the version used for the paper).
"""

import os

import numpy as np
import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import ReproducibleRandomSampler

from .constants import SER_LABELS
from .ser import compute_metrics, format_metrics

# Seed of SpeechBrain's ReproducibleRandomSampler when ``shuffle: True`` is given
# without a sampler. Stage 1 and repetition 0 of Stage 3 were run with it.
DEFAULT_SAMPLER_SEED = 563375142


class SERBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Task adapter on latents of Z^(layer), followed by a log-softmax."""
        batch = batch.to(self.device)
        feats, _ = batch.feat
        outputs = self.modules.output_mlp(feats, self.hparams.layer)
        return self.hparams.log_softmax(outputs)

    def compute_objectives(self, predictions, batch, stage):
        emoid, _ = batch.emo_encoded
        emoid = emoid.squeeze(1)  # to meet the input form of nll loss
        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append((predictions.argmax(-1).cpu().numpy(), emoid.cpu().numpy()))
        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.error_metrics = []

    def _next_learning_rate(self):
        new_lr = self.hparams.lr_annealing.get_next_value()
        sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
        return new_lr

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            if not self.has_valid_set:  # no validation stage: step the scheduler here
                new_lr = self._next_learning_rate()
                self.hparams.train_logger.log_stats({"Epoch": epoch, "lr": new_lr},
                                                    train_stats={"loss": self.train_loss})
                self._checkpoint(epoch, {"loss": float(stage_loss)})
            return

        pred = np.concatenate([p for p, _ in self.error_metrics])
        target = np.concatenate([t for _, t in self.error_metrics])
        metrics = compute_metrics(target, pred)
        stats = {"loss": stage_loss, "wa": metrics["WA"], "ua": metrics["UA"], "macro_f1": metrics["macro_F1"]}
        if stage == sb.Stage.VALID:
            new_lr = self._next_learning_rate()
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats({"Epoch": epoch, "lr": new_lr},
                                                train_stats={"loss": self.train_loss}, valid_stats=stats)
            self._checkpoint(epoch, stats)
            if epoch % self.hparams.ckpt_interval_epochs == 0:
                print(f"epoch {epoch:3d}  train loss {self.train_loss:.4f}  valid {format_metrics(metrics)}", flush=True)

    def _checkpoint(self, epoch, stats):
        if epoch % self.hparams.ckpt_interval_epochs == 0:
            self.checkpointer.save_and_keep_only(meta={k: float(v) for k, v in stats.items()}, num_to_keep=1)

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)


def make_dataset(items, latents):
    """SpeechBrain dataset over utterance-level latents.

    ``items`` maps an utterance id to ``{"emo": ...}`` (the order of the dict is
    the order the sampler indexes) and ``latents`` maps the same ids to vectors.
    """
    # The label encoder assigns the SER labels their index in alphabetical order.
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    for label in SER_LABELS:
        label_encoder.add_label(label)
    label_encoder.expect_len(len(SER_LABELS))

    @sb.utils.data_pipeline.takes("id")
    @sb.utils.data_pipeline.provides("feat")
    def feature_pipeline(id):
        return torch.FloatTensor(latents[id])

    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo_encoded")
    def label_pipeline(emo):
        return label_encoder.encode_label_torch(emo)

    data = {key: {"emo": item["emo"]} for key, item in items.items()}
    return sb.dataio.dataset.DynamicItemDataset(
        data, dynamic_items=[feature_pipeline, label_pipeline], output_keys=["id", "feat", "emo_encoded"])


def train(hparams_file, output_folder, train_items, train_latents, valid_items=None, valid_latents=None,
          layer=0, epochs=100, lr=1e-3, batch_size=32, seed=1234, sampler_seed=DEFAULT_SAMPLER_SEED,
          init_adapter=None, start_epoch=0, device="cuda"):
    """Run the recipe and return the trained :class:`~gelda.adapter.TaskAdapter`.

    Stage 1: ``layer=0`` and ``init_adapter=None`` (epochs 1..``epochs``).
    Stage 3: ``init_adapter`` is the Stage-1 adapter and ``start_epoch`` the
    number of epochs it was trained for. Its layers producing Z^(<=layer) are
    frozen and training continues for ``epochs`` more epochs (101-200 in the
    paper) with a new optimizer and a new learning-rate schedule whose length
    is the total number of epochs (``start_epoch + epochs``), as in the paper.
    """
    encoder_dim = int(np.asarray(next(iter(train_latents.values()))).shape[-1]) if init_adapter is None \
        else init_adapter.in_dim
    overrides = {
        "output_folder": output_folder, "seed": seed, "layer": layer, "encoder_dim": encoder_dim,
        "number_of_epochs": start_epoch + epochs, "lr": lr, "batch_size": batch_size,
    }
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.create_experiment_directory(experiment_directory=output_folder, hyperparams_to_save=hparams_file,
                                   overrides=overrides)

    brain = SERBrain(modules=hparams["modules"], opt_class=hparams["opt_class"], hparams=hparams,
                     run_opts={"device": device, "noprogressbar": True}, checkpointer=hparams["checkpointer"])
    adapter = hparams["output_mlp"]
    if init_adapter is not None:
        adapter.load_state_dict(init_adapter.state_dict())
        adapter.freeze_up_to(layer)
        hparams["epoch_counter"].current = start_epoch

    train_set = make_dataset(train_items, train_latents)
    valid_set = make_dataset(valid_items, valid_latents) if valid_items else None
    brain.has_valid_set = valid_set is not None

    # `shuffle: True` in SpeechBrain is a ReproducibleRandomSampler that is re-seeded
    # with (seed + epoch) before every epoch.
    train_loader_kwargs = dict(hparams["dataloader_options"])
    if train_loader_kwargs.pop("shuffle", False):
        brain.train_sampler = ReproducibleRandomSampler(train_set, seed=sampler_seed)
        train_loader_kwargs["sampler"] = brain.train_sampler
    valid_loader_kwargs = dict(hparams["dataloader_options"], shuffle=False)

    brain.fit(epoch_counter=hparams["epoch_counter"], train_set=train_set, valid_set=valid_set,
              train_loader_kwargs=train_loader_kwargs, valid_loader_kwargs=valid_loader_kwargs)
    return adapter.eval()


def default_hparams():
    """``configs/ser/adapter.yaml`` of this repository."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "ser", "adapter.yaml")
