"""FedMetro's gated graph recurrence adapted to PeMS client partitions.

Only shared graph-convolution pools, polynomial coefficients, and prediction
heads are averaged. Historical encoders, masks, and node embeddings stay local.
"""

import copy
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


def train_fedmetro_paper(ctx, args, get_setting, log_experiment_results):
    if args.protection != 'plain':
        raise ValueError('The gated FedMetro trainer currently supports plain communication only')

    def as_list(value):
        return value if isinstance(value, list) else [value]

    def broadcast(tag, value, hosts):
        ctx.guest.put(tag, value)
        ctx.hosts.put(tag, [value] * hosts)

    tags = [f't{step}_{gate}' for step in range(args.t_in) for gate in ('gate', 'update')]
    communication_bytes = 0

    if not ctx.is_on_arbiter:
        train, val, test, model, optimizer, loss_fn, _, _, _, scaler, _ = get_setting(ctx)
        model.set_hyper_source(train.tensors[0])
        loaders = {
            'train': DataLoader(train, batch_size=args.batch_size, shuffle=True),
            'val': DataLoader(val, batch_size=args.batch_size, shuffle=False),
            'test': DataLoader(test, batch_size=args.batch_size, shuffle=False),
        }
        ctx.arbiter.put('fedmetro_loader_sizes', {part: len(loader) for part, loader in loaders.items()})
        best_state = None
        best_epoch = -1
        best_val = float('inf')
        total_train_time = total_val_time = 0.0
        rounds = 0
    else:
        sizes = [ctx.guest.get('fedmetro_loader_sizes')] + as_list(ctx.hosts.get('fedmetro_loader_sizes'))
        steps = {part: min(size[part] for size in sizes) for part in ('train', 'val', 'test')}
        if getattr(args, 'fedmetro_max_steps', 0) > 0:
            steps = {part: min(value, args.fedmetro_max_steps) for part, value in steps.items()}
        broadcast('fedmetro_steps', steps, len(sizes) - 1)
        best_global = float('inf')
        stale = 0
    if not ctx.is_on_arbiter:
        steps = ctx.arbiter.get('fedmetro_steps')
        if isinstance(steps, list):
            steps = steps[0]

    def client_exchange(phase, local_tag, tensor, prefix):
        nonlocal communication_bytes
        wire_tag = f'{prefix}_{phase}_{local_tag}'
        payload = tensor.detach().cpu().contiguous()
        communication_bytes += payload.numel() * payload.element_size()
        ctx.arbiter.put(wire_tag, payload)
        reply = ctx.arbiter.get(f'{wire_tag}_sum')
        if isinstance(reply, list):
            reply = reply[0]
        communication_bytes += reply.numel() * reply.element_size()
        return reply.to(tensor.device, dtype=tensor.dtype)

    def serve_batch(prefix, training):
        for phase, ordered in [('forward', tags), ('backward', reversed(tags))] if training else [('forward', tags)]:
            for local_tag in ordered:
                wire_tag = f'{prefix}_{phase}_{local_tag}'
                values = [ctx.guest.get(wire_tag)] + as_list(ctx.hosts.get(wire_tag))
                total = sum(values)
                broadcast(f'{wire_tag}_sum', total, len(values) - 1)

    def evaluate(part, epoch=None):
        model.eval()
        absolute = squared = mape_numerator = 0.0
        elements = mape_elements = 0
        with torch.no_grad():
            for step, (x, y) in enumerate(loaders[part]):
                if step >= steps[part]:
                    break
                prefix = f'fedmetro_{part}_{epoch}_{step}'
                x = x.to(args.device)
                pred, _ = model(x, lambda phase, tag, tensor: client_exchange(phase, tag, tensor, prefix))
                target = y.to(args.device).squeeze(-1)
                if part == 'test':
                    pred = scaler.inverse_transform(pred)
                    target = scaler.inverse_transform(target)
                difference = pred - target
                absolute += difference.abs().sum().item()
                squared += difference.square().sum().item()
                elements += difference.numel()
                if part == 'test':
                    valid = target > 0.5
                    mape_numerator += (difference[valid].abs() / target[valid]).sum().item()
                    mape_elements += valid.sum().item()
        return absolute, squared, elements, mape_numerator, mape_elements

    for epoch in range(args.epochs):
        if not ctx.is_on_arbiter:
            rounds = epoch + 1
            model.train()
            started = time.time()
            for step, (x, y) in enumerate(loaders['train']):
                if step >= steps['train']:
                    break
                prefix = f'fedmetro_train_{epoch}_{step}'
                optimizer.zero_grad()
                pred, regularizer = model(
                    x.to(args.device),
                    lambda phase, tag, tensor: client_exchange(phase, tag, tensor, prefix),
                )
                target = y.to(args.device).squeeze(-1)
                (loss_fn(pred, target) + regularizer).backward()
                if getattr(args, 'fedmetro_grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.fedmetro_grad_clip)
                optimizer.step()
            total_train_time += time.time() - started
            shared_names = ('gate.', 'update.', 'poly_coefficients', 'prediction_head.')
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                     if key.startswith(shared_names)}
            communication_bytes += sum(value.numel() * value.element_size() for value in state.values())
            ctx.arbiter.put(f'fedmetro_shared_{epoch}', state)
            global_state = ctx.arbiter.get(f'fedmetro_global_{epoch}')
            if isinstance(global_state, list):
                global_state = global_state[0]
            communication_bytes += sum(value.numel() * value.element_size() for value in global_state.values())
            model.load_state_dict(global_state, strict=False)
            # The author's FED trainer recreates Adam after each FedAvg round.
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8,
                                         weight_decay=0, amsgrad=False)
            started = time.time()
            val_stats = evaluate('val', epoch)
            total_val_time += time.time() - started
            val_mae = val_stats[0] / max(val_stats[2], 1)
            if val_mae < best_val:
                best_val = val_mae
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            ctx.arbiter.put(f'fedmetro_val_{epoch}', val_mae)
            stop = ctx.arbiter.get(f'fedmetro_stop_{epoch}')
            if isinstance(stop, list):
                stop = stop[0]
            if stop:
                break
        else:
            for step in range(steps['train']):
                serve_batch(f'fedmetro_train_{epoch}_{step}', training=True)
            states = [ctx.guest.get(f'fedmetro_shared_{epoch}')] + as_list(ctx.hosts.get(f'fedmetro_shared_{epoch}'))
            averaged = {key: sum(state[key] for state in states) / len(states) for key in states[0]}
            broadcast(f'fedmetro_global_{epoch}', averaged, len(states) - 1)
            for step in range(steps['val']):
                serve_batch(f'fedmetro_val_{epoch}_{step}', training=False)
            scores = [ctx.guest.get(f'fedmetro_val_{epoch}')] + as_list(ctx.hosts.get(f'fedmetro_val_{epoch}'))
            score = sum(scores) / len(scores)
            if score < best_global - 1e-4:
                best_global, stale = score, 0
            else:
                stale += 1
            stop = stale >= 50
            broadcast(f'fedmetro_stop_{epoch}', stop, len(states) - 1)
            if stop:
                break

    if ctx.is_on_arbiter:
        for step in range(steps['test']):
            serve_batch(f'fedmetro_test_None_{step}', training=False)
        return

    if best_state is not None:
        model.load_state_dict(best_state)
    started = time.time()
    absolute, squared, elements, mape_sum, mape_elements = evaluate('test')
    elapsed = time.time() - started
    mae = absolute / max(elements, 1)
    mse = squared / max(elements, 1)
    log_experiment_results(
        model_name='FedMetro', dataset_client=f'{args.dataset_name}_client{ctx.rank}',
        feature_type=args.feature_type, best_epoch=best_epoch + 1,
        acc_mae=round(mae, 4), acc_mse=round(mse, 4), acc_rmse=round(math.sqrt(mse), 4),
        acc_mape=round(100 * mape_sum / max(mape_elements, 1), 4),
        acc_elements=elements, acc_abs_error_sum=round(absolute, 4),
        acc_sq_error_sum=round(squared, 4), acc_mape_error_sum=round(mape_sum, 4),
        acc_mape_elements=mape_elements, eff_train_time=round(total_train_time, 2),
        eff_val_time=round(total_val_time / max(rounds, 1), 4),
        eff_test_time=round(elapsed, 4),
        eff_comm_size_mb=round(communication_bytes / (1024 * 1024), 4),
        eff_train_round=rounds, eff_flops=float('nan'), dp_noise=0.0,
    )
