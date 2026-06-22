import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
from torch.func import vmap, jacrev
from concurrent.futures import ThreadPoolExecutor
import threading
import pdb

_compile_lock = threading.Lock()

BATCH_SIZE = 64
N_REPLICATES = 4
JOBS_PER_GPU = 8   # concurrent training runs per GPU for weight-scale experiment
SHOW_PLOTS = False  # if False, save figures as PDFs instead of displaying
USE_ROPE = True # if False, use learned position encoding.

if USE_ROPE:
	from rotary_embedding_torch import RotaryEmbedding

if not SHOW_PLOTS:
    matplotlib.use('Agg')  # thread-safe non-interactive backend
import matplotlib.pyplot as plt

def show_or_save(fig, name):
	if SHOW_PLOTS:
		plt.show()
	else:
		rope = 'rope' if USE_ROPE else ''
		fname = f"fastgrok_{rope}_{name}.pdf"
		fig.savefig(fname, bbox_inches='tight')
		print(f"Saved {fname}")
		plt.close(fig)


class GrokkingTransformer(nn.Module):
	def __init__(self, p, d, n_heads=1, use_norm=False, mlp_bias=True, mlp_act='relu'):
		super().__init__()
		assert d % n_heads == 0, "d must be divisible by n_heads"
		self.p = p
		self.d = d
		self.n_heads = n_heads
		self.head_dim = d // n_heads

		# Vocab: 0 to p-1 are standard numbers. Token `p` is the special '=' token.
		self.embed = nn.Embedding(p + 1, d)
		nn.init.normal_(self.embed.weight, std=0.02)

		if USE_ROPE:
			self.rope = RotaryEmbedding(dim=self.head_dim)
		else:
			self.pos_emb = nn.Embedding(3, d)
			nn.init.normal_(self.pos_emb.weight, std=0.02)

		# Multi-head attention projections
		self.W_q = nn.Linear(d, d, bias=False)
		self.W_k = nn.Linear(d, d, bias=False)
		self.W_v = nn.Linear(d, d, bias=False)
		self.W_o = nn.Linear(d, d, bias=False)

		self.ln1 = nn.LayerNorm(d) if use_norm else nn.Identity()
		self.ln2 = nn.LayerNorm(d) if use_norm else nn.Identity()

		self.mlp_w1 = nn.Linear(d, 4 * d, bias=mlp_bias)
		self.mlp_w2 = nn.Linear(4 * d, d, bias=mlp_bias)
		self.mlp_act = mlp_act

		self.W_out = nn.Linear(d, p, bias=True)
		nn.init.normal_(self.W_out.weight, std=1.0 / np.sqrt(d))

	def forward(self, x):
		# x shape: (B, 3)
		e = self.embed(x)
		return self.forward_from_embeddings(e)

	def forward_from_embeddings(self, e):
		# Unroll logic allows torch.func.vmap to run over the sequence
		# pdb.set_trace()
		is_batched = e.dim() == 3
		if not is_batched: e = e.unsqueeze(0)

		B, seq_len, d = e.shape

		if not USE_ROPE:
			positions = torch.arange(seq_len, device=e.device)
			# positions = torch.tensor([0, 0, 2], device=e.device)
			# force commutativity.  This is worse!
			e = e + self.pos_emb(positions)

		x_norm = self.ln1(e)
		q, k, v = self.W_q(x_norm), self.W_k(x_norm), self.W_v(x_norm)

		# Split into heads: (B, seq, d) -> (B*n_heads, seq, head_dim)
		def split_heads(t):
			return t.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2).reshape(B * self.n_heads, seq_len, self.head_dim)

		q, k, v = split_heads(q), split_heads(k), split_heads(v)

		if USE_ROPE:
			q = self.rope.rotate_queries_or_keys(q.view(B, self.n_heads, seq_len, self.head_dim)).reshape(B * self.n_heads, seq_len, self.head_dim)
			k = self.rope.rotate_queries_or_keys(k.view(B, self.n_heads, seq_len, self.head_dim)).reshape(B * self.n_heads, seq_len, self.head_dim)

		scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.head_dim)

		# Standard causal mask for autoregressive emulation
		# mask not strictly required..
		mask = torch.tril(torch.ones(seq_len, seq_len, device=e.device)).unsqueeze(0)
		scores = scores.masked_fill(mask == 0, float('-inf'))
		attn = torch.softmax(scores, dim=-1)

		# Merge heads back: (B*n_heads, seq, head_dim) -> (B, seq, d)
		context = torch.bmm(attn, v).view(B, self.n_heads, seq_len, self.head_dim).transpose(1, 2).reshape(B, seq_len, d)
		h = e + context

		# MLP block
		h_norm = self.ln2(context)
		h_mid = self.mlp_w1(h_norm)
		h_mid = h_mid ** 2 if self.mlp_act == 'quadratic' else nn.functional.relu(h_mid)
		mlp_out = self.mlp_w2(h_mid)
		h = h + mlp_out

		# We only predict from the last token '=' (position index 2)
		logits = self.W_out(h[:, -1, :])

		if not is_batched: logits = logits.squeeze(0)
		return logits


def make_device_selector():
	n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
	if n_gpus > 1:
		print(f"Found {n_gpus} GPUs, cycling replicates across them.")
		return lambda rep: torch.device(f'cuda:{rep % n_gpus}')
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Using device: {device}")
		return lambda rep: device


def train_model(p=59, d=64, epochs=250, device='cpu', batch_size=None, use_norm=False, mlp_bias=True, weight_decay=2e-2, train_frac=0.6, mlp_act='relu', betas=(0.9, 0.995), schedule_wd=False, weight_scale=1.0, compile_model=True):
	dataset, labels = [], []
	for a in range(p):
		for b in range(p):
			dataset.append([a, b, p]) # p is the '=' token index
			labels.append((a + b) % p)

	dataset, labels = torch.tensor(dataset, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

	indices = torch.randperm(p * p)
	split_idx = int(train_frac * p * p)
	train_idx, val_idx = indices[:split_idx], indices[split_idx:]

	train_data, val_data = dataset[train_idx].to(device), dataset[val_idx].to(device)
	train_labels, val_labels = labels[train_idx].to(device), labels[val_idx].to(device)

	n_train = train_data.shape[0]
	use_minibatch = batch_size is not None and batch_size < n_train

	# steps_per_epoch: gradient steps per effective epoch (full-dataset pass equivalent)
	steps_per_epoch = n_train // batch_size if use_minibatch else 1
	total_steps = epochs * steps_per_epoch
	# For small batches, cap validation frequency to ~1 check per 256 samples processed
	if use_minibatch and batch_size < 256:
		log_every = max(1, 256 // batch_size)
	else:
		log_every = steps_per_epoch

	model = GrokkingTransformer(p, d, use_norm=use_norm, mlp_bias=mlp_bias, mlp_act=mlp_act).to(device)

	if weight_scale != 1.0:
		with torch.no_grad():
			for param in model.parameters():
				param.mul_(weight_scale)

	optimizer = optim.AdamW(model.parameters(),
							lr=1e-3,
							weight_decay=weight_decay,
							amsgrad=True,
							betas=betas,
							eps=1e-5)              #  prevent division-by-zero explosions)
	criterion = nn.CrossEntropyLoss()

	if compile_model:
		model = torch.compile(model)
		# CompileEventLogger.compilation_metric is called on every compiled
		# backward (not just first-time compilation) and writes to shared
		# class-level state — it is not thread-safe. Pre-warm fwd+bwd under a
		# lock so all AOT-Autograd tracing finishes before parallel training
		# begins. Only safe to use compile_model=True in single-threaded runs.
		with _compile_lock:
			_nb = min(batch_size if batch_size is not None else n_train, n_train)
			criterion(model(train_data[:_nb]), train_labels[:_nb]).backward()
			optimizer.zero_grad()
			model(val_data)

	history = {'train_loss':[], 'val_loss':[], 'val_acc':[], 'epoch_x':[]}
	current_wd = weight_decay
	current_lr = 1e-3
	last_decay_epoch = -1

	for step in range(total_steps):
		optimizer.zero_grad()

		if use_minibatch:
			mb_idx = torch.randperm(n_train, device=device)[:batch_size]
			batch_data = train_data[mb_idx]
			batch_labels = train_labels[mb_idx]
		else:
			batch_data = train_data
			batch_labels = train_labels

		if schedule_wd:
			epoch_num = step // steps_per_epoch
			if epoch_num >= 24 and (epoch_num - 24) % 8 == 0 and epoch_num != last_decay_epoch:
				current_wd *= 0.5
				current_lr *= 0.5
				last_decay_epoch = epoch_num
				for pg in optimizer.param_groups:
					pg['weight_decay'] = current_wd
					pg['learning_rate'] = current_lr

		logits = model(batch_data)
		loss = criterion(logits, batch_labels)
		loss.backward()
		optimizer.step()

		if step % log_every == 0:
			# No train/eval toggle and no no_grad: avoids dynamo grad_mode recompilation.
			# This model has no dropout/batchnorm so toggling is a no-op anyway.
			# .detach() frees the val graph immediately without building a second compiled variant.
			val_logits = model(val_data).detach()
			v_loss = criterion(val_logits, val_labels).item()
			preds = torch.argmax(val_logits, dim=1)
			v_acc = (preds == val_labels).float().mean().item()

			history['epoch_x'].append(step / steps_per_epoch)
			history['train_loss'].append(loss.item())
			history['val_loss'].append(v_loss)
			history['val_acc'].append(v_acc)

			if v_acc >= 1.0:
					break

	return model, history


def run_experiment():
	get_device = make_device_selector()
	epochs = 250
	n_replicates = N_REPLICATES

	conditions = [
		dict(label='No LayerNorm', color='blue',   train_color='cornflowerblue', use_norm=False, mlp_bias=True,  weight_decay=2e-2, mlp_act='relu'),
		dict(label='LayerNorm', color='orange', train_color='sandybrown',     use_norm=True, mlp_bias=True, weight_decay=2e-2, mlp_act='relu'),
		dict(label='LayerNorm, no MLP bias', color='green',  train_color='limegreen', use_norm=True,  mlp_bias=False, weight_decay=2e-2, mlp_act='relu'),
		dict(label='LayerNorm, no bias, no decay', color='red',    train_color='lightsalmon', use_norm=True, mlp_bias=False, weight_decay=0.0, mlp_act='relu'),
	]

	# results[cond_idx] = list of (model, history) across replicates
	results = [None for _ in conditions]

	def train_condition(ci_cond):
		ci, cond = ci_cond
		def train_rep(rep):
			dev = get_device(rep)
			print(f"Training {cond['label']} replicate {rep+1}/{n_replicates} on {dev}...")
			return train_model(
				epochs=epochs, device=dev, batch_size=BATCH_SIZE,
				use_norm=cond['use_norm'], mlp_bias=cond['mlp_bias'],
				weight_decay=cond['weight_decay'], mlp_act=cond['mlp_act'])
		with ThreadPoolExecutor(max_workers=n_replicates) as pool:
			return list(pool.map(train_rep, range(n_replicates)))

	for ci, cond in enumerate(conditions):
		results[ci] = train_condition((ci, cond))

	# Plot 1: Curves
	fig1, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

	for ci, cond in enumerate(conditions):
		for rep, (_, hist) in enumerate(results[ci]):
			label = cond['label'] if rep == 0 else '_nolegend_'
			ax_acc.plot(hist['epoch_x'], hist['val_acc'], label=label, color=cond['color'], alpha=0.6, marker='o', markevery=[-1], markersize=5)
			ax_loss.plot(hist['epoch_x'], hist['train_loss'], label=f"{cond['label']} Train" if rep == 0 else '_nolegend_',
				color=cond['train_color'], linestyle='--', alpha=0.6, marker='o', markevery=[-1], markersize=5)
			ax_loss.plot(hist['epoch_x'], hist['val_loss'], label=f"{cond['label']} Val" if rep == 0 else '_nolegend_',
				color=cond['color'], alpha=0.6, marker='o', markevery=[-1], markersize=5)

	ax_acc.set_title("Validation Accuracy")
	ax_acc.set_xlabel("Effective Epochs")
	ax_acc.set_ylabel("Accuracy")
	ax_acc.legend()
	ax_acc.grid(True, alpha=0.3)

	ax_loss.set_title("Loss")
	ax_loss.set_xlabel("Effective Epochs")
	ax_loss.set_ylabel("Loss")
	ax_loss.legend()
	ax_loss.grid(True, alpha=0.3)

	plt.tight_layout()
	show_or_save(fig1, 'curves')

	# Plot 2: AGOP kernel grid — rows=replicates, cols=conditions
	def get_agop_matrix(model):
		model.eval()
		p = model.p
		dev = next(model.parameters()).device
		dataset = torch.tensor([[a, b, p] for a in range(p) for b in range(p)]).to(dev)

		e = model.embed(dataset)
		J = vmap(jacrev(lambda x: model.forward_from_embeddings(x)))(e)

		J_a = J[:, :, 0, :]
		J_a_centered = J_a - J_a.mean(dim=0, keepdim=True)
		M_a = torch.einsum('bpi,bpj->ij', J_a_centered, J_a_centered) / dataset.shape[0]

		E_num = model.embed.weight[:-1]
		G_A = (E_num @ M_a @ E_num.t()).cpu().detach().numpy()
		return G_A

	n_conds = len(conditions)
	fig2, axes = plt.subplots(n_replicates, n_conds, figsize=(6 * n_conds, 5 * n_replicates))

	for rep in range(n_replicates):
		for ci, cond in enumerate(conditions):
			ax = axes[rep][ci]
			model = results[ci][rep][0]
			im = ax.imshow(get_agop_matrix(model), cmap='magma')
			title = cond['label'] if rep == 0 else ''
			rep_label = f"Rep {rep+1}"
			ax.set_title(f"{title}\n{rep_label}" if title else rep_label)
			plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

	plt.tight_layout()
	show_or_save(fig2, 'agop')

def run_experiment_train_frac():
	get_device = make_device_selector()
	epochs = 250
	n_replicates = N_REPLICATES
	train_fracs = [0.30, 0.40, 0.50, 0.70]
	colors      = ['purple', 'blue', 'orange', 'green']
	train_colors= ['plum',   'cornflowerblue', 'sandybrown', 'limegreen']

	# results[frac_idx] = list of (model, history) across replicates
	results = [None for _ in train_fracs]

	for fi, frac in enumerate(train_fracs):
		def train_rep(rep, frac=frac):
			dev = get_device(rep)
			print(f"Training p=113 train_frac={frac:.0%} replicate {rep+1}/{n_replicates} on {dev}...")
			return train_model(
				p=113, epochs=epochs, device=dev, batch_size=8, #note smaller batch size!
				use_norm=True, mlp_bias=False, weight_decay=0.02, train_frac=frac, schedule_wd=True)
		with ThreadPoolExecutor(max_workers=n_replicates) as pool:
			results[fi] = list(pool.map(train_rep, range(n_replicates)))

	fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
	fig.suptitle("p=113, LayerNorm on, no MLP bias — varying train fraction")

	for fi, frac in enumerate(train_fracs):
		for rep, (_, hist) in enumerate(results[fi]):
			label = f"{frac:.0%} train" if rep == 0 else '_nolegend_'
			ax_acc.plot(hist['epoch_x'], hist['val_acc'], label=label, color=colors[fi], alpha=0.6, marker='o', markevery=[-1], markersize=5)
			ax_loss.plot(hist['epoch_x'], hist['train_loss'],
				label=f"{frac:.0%} Train" if rep == 0 else '_nolegend_',
				color=train_colors[fi], linestyle='--', alpha=0.6, marker='o', markevery=[-1], markersize=5)
			ax_loss.plot(hist['epoch_x'], hist['val_loss'],
				label=f"{frac:.0%} Val" if rep == 0 else '_nolegend_',
				color=colors[fi], alpha=0.6, marker='o', markevery=[-1], markersize=5)

	ax_acc.set_title("Validation Accuracy")
	ax_acc.set_xlabel("Effective Epochs")
	ax_acc.set_ylabel("Accuracy")
	ax_acc.legend()
	ax_acc.grid(True, alpha=0.3)

	ax_loss.set_title("Loss")
	ax_loss.set_xlabel("Effective Epochs")
	ax_loss.set_ylabel("Loss")
	ax_loss.legend()
	ax_loss.grid(True, alpha=0.3)

	plt.tight_layout()
	show_or_save(fig, 'train_frac')

def run_experiment_batch_size():
	get_device = make_device_selector()
	epochs = 250
	n_replicates = N_REPLICATES
	p = 59
	train_frac = 0.6
	n_train = int(train_frac * p * p)  # 2088

	# Powers of 2 from 8 up to n_train, then None (full batch)
	batch_sizes = [2**k for k in range(0, 12) if 2**k < n_train] + [None]
	labels = [str(bs) if bs is not None else f"full ({n_train})" for bs in batch_sizes]

	cmap = matplotlib.colormaps['hsv']
	colors = [(r*0.7, g*0.7, b*0.7, a) for r, g, b, a in [cmap(i / (len(batch_sizes) - 1)) for i in range(len(batch_sizes))]]

	results = [None for _ in batch_sizes]

	for bi, bs in enumerate(batch_sizes):
		def train_rep(rep, bs=bs):
			dev = get_device(rep)
			print(f"Training batch_size={bs} replicate {rep+1}/{n_replicates} on {dev}...")
			return train_model(
				p=p, epochs=epochs, device=dev, batch_size=bs,
				use_norm=True, mlp_bias=False, weight_decay=0.02, train_frac=train_frac)
		with ThreadPoolExecutor(max_workers=n_replicates) as pool:
			results[bi] = list(pool.map(train_rep, range(n_replicates)))

	fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
	fig.suptitle("p=59, LN on, no bias, 60% train frac")

	for bi, (bs, label) in enumerate(zip(batch_sizes, labels)):
		for rep, (_, hist) in enumerate(results[bi]):
			lbl = label if rep == 0 else '_nolegend_'
			ax_acc.plot(hist['epoch_x'], hist['val_acc'], label=lbl, color=colors[bi], alpha=0.6, marker='o', markevery=[-1], markersize=5)
			ax_loss.plot(hist['epoch_x'], hist['train_loss'],
				label=f"{label} train" if rep == 0 else '_nolegend_',
				color=colors[bi], linestyle='--', alpha=0.6, marker='o', markevery=[-1], markersize=5)
			ax_loss.plot(hist['epoch_x'], hist['val_loss'],
				label=f"{label} val" if rep == 0 else '_nolegend_',
				color=colors[bi], alpha=0.6, marker='o', markevery=[-1], markersize=5)

	ax_acc.set_title("Validation Accuracy")
	ax_acc.set_xlabel("Effective Epochs")
	ax_acc.set_ylabel("Accuracy")
	ax_acc.legend(fontsize=7)
	ax_acc.grid(True, alpha=0.3)

	ax_loss.set_title("Loss")
	ax_loss.set_xlabel("Effective Epochs")
	ax_loss.set_ylabel("Loss")
	ax_loss.legend(fontsize=7)
	ax_loss.grid(True, alpha=0.3)

	plt.tight_layout()
	show_or_save(fig, 'batch_size')

def run_experiment_weight_scale():
	epochs = 250
	n_replicates = N_REPLICATES
	p = 113
	train_frac = 0.7
	batch_size = 16

	n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
	base_device = 'cuda' if torch.cuda.is_available() else 'cpu'
	max_workers = n_gpus * JOBS_PER_GPU  # e.g. 2 GPUs × 8 = 16 concurrent jobs

	# 3 points per octave from 1/16 (2^-4) to 16 (2^4), inclusive → 8 octaves × 3 + 1 = 25 points
	scales = np.geomspace(1/16, 16, 3 * 8 + 1)

	cmap = matplotlib.colormaps['plasma']
	colors = [cmap(i / (len(scales) - 1)) for i in range(len(scales))]

	# Build a flat list of (scale_idx, scale, rep) jobs; device cycles across GPUs by job slot
	all_jobs = [(si, scale, rep)
	            for si, scale in enumerate(scales)
	            for rep in range(n_replicates)]

	def train_job(args):
		si, scale, rep = args
		job_idx = si * n_replicates + rep
		dev = torch.device(f'cuda:{job_idx % n_gpus}') if n_gpus > 1 else torch.device(base_device)
		print(f"Training weight_scale={scale:.4f} rep {rep+1}/{n_replicates} on {dev}...")
		return train_model(
			p=p, epochs=epochs, device=dev, batch_size=batch_size,
			use_norm=True, mlp_bias=False, weight_decay=0.02,
			train_frac=train_frac, weight_scale=scale,
			compile_model=False)  # torch.compile is not thread-safe (CompileEventLogger race)

	print(f"Running {len(all_jobs)} jobs across {n_gpus} GPU(s), {max_workers} concurrent.")
	with ThreadPoolExecutor(max_workers=max_workers) as pool:
		flat_results = list(pool.map(train_job, all_jobs))

	# Reassemble into results[si][rep]
	results = [[None] * n_replicates for _ in scales]
	for (si, scale, rep), result in zip(all_jobs, flat_results):
		results[si][rep] = result

	# Plot 1: Learning curves (val acc + train/val loss)
	fig1, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
	fig1.suptitle(f"p={p}, LN on, no MLP bias, {train_frac:.0%} train — varying initial weight scale")

	for si, scale in enumerate(scales):
		label = f"×{scale:.3g}"
		for rep, (_, hist) in enumerate(results[si]):
			lbl = label if rep == 0 else '_nolegend_'
			ax_acc.plot(hist['epoch_x'], hist['val_acc'],
				label=lbl, color=colors[si], alpha=0.6, marker='o', markevery=[-1], markersize=5)
			ax_loss.plot(hist['epoch_x'], hist['train_loss'],
				label=f"{label} train" if rep == 0 else '_nolegend_',
				color=colors[si], linestyle='--', alpha=0.6, marker='o', markevery=[-1], markersize=5)
			ax_loss.plot(hist['epoch_x'], hist['val_loss'],
				label=f"{label} val" if rep == 0 else '_nolegend_',
				color=colors[si], alpha=0.6, marker='o', markevery=[-1], markersize=5)

	ax_acc.set_title("Validation Accuracy")
	ax_acc.set_xlabel("Effective Epochs")
	ax_acc.set_ylabel("Accuracy")
	ax_acc.legend(fontsize=6, ncol=2)
	ax_acc.grid(True, alpha=0.3)

	ax_loss.set_title("Loss")
	ax_loss.set_xlabel("Effective Epochs")
	ax_loss.set_ylabel("Loss")
	ax_loss.legend(fontsize=6, ncol=2)
	ax_loss.grid(True, alpha=0.3)

	plt.tight_layout()
	show_or_save(fig1, 'weight_scale_curves')

	# Plot 2: Summary — final val acc (mean ± std across replicates) vs scale on a log x-axis
	fig2, ax = plt.subplots(figsize=(8, 5))
	fig2.suptitle(f"p={p}, LN on, no MLP bias, {train_frac:.0%} train — final val acc vs weight scale")

	final_means = []
	final_stds = []
	for si, scale in enumerate(scales):
		accs = [hist['val_acc'][-1] for _, hist in results[si]]
		final_means.append(np.mean(accs))
		final_stds.append(np.std(accs))

	ax.errorbar(scales, final_means, yerr=final_stds, fmt='o-', color='steelblue', capsize=4, linewidth=1.5)
	ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='default scale (×1)')
	ax.set_xscale('log', base=2)
	ax.set_xlabel("Weight Scale Multiplier (log₂ scale)")
	ax.set_ylabel("Final Validation Accuracy")
	ax.set_title("Final Validation Accuracy vs Initial Weight Scale")
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	show_or_save(fig2, 'weight_scale_summary')


if __name__ == "__main__":
	# run_experiment()
	# run_experiment_train_frac()
	# run_experiment_batch_size()
	run_experiment_weight_scale()
