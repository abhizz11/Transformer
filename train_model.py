from __future__ import annotations

import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Memory and Speed Optimization
def _autocast_context(device: torch.device):
    '''Training LLMs in standard-32 bit floats takes up too much VRAM and is relatively slow, 
    that's why we are using FP16 which cuts memory usage in half and utilizes fast Tensor Cores'''
    if device.type == "cuda":
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()

# Function that returns average token_level cross-entropy on validation data
@torch.inference_mode()
def evaluate_model(
        model, 
        data_loader,
        device: torch.device,
        max_batches: int | None = 50,
):
    '''We start with inference mode to shut down the gradient-tracking. Making the forward pass significantly cheaper and faster.
    We also do model.eval() so that layers like Dropout are turned off'''
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    try:
        for batch_index, (input_batch, target_batch) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break 
            input_batch = input_batch.to(device, non_blocking = True)
            target_batch = target_batch.to(device, non_blocking = True)

            with _autocast_context(device):
                logits = model(input_batch)
                loss_sum = F.cross_entropy(
                    logits.flatten(0, 1),
                    target_batch.flatten(),
                    reduction="sum",
                )

            total_loss += loss_sum.item()
            total_tokens += target_batch.numel()
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("Validation loader produced no tokens.")

    return total_loss / total_tokens                

# AdamW function that excludes biases and normalization vectors from decay
def create_optimizer(
        model,
        learning_rate = 3e-4,
        weight_decay = 0.1,
        betas = (0.9, 0.95),
):
    '''Standard weight decay(L2) shrinks all model weights slightly during every step to prevent overfitting. This function 
    explicitly loops through all model parameters and splits them into two lists. Biases and normalization should not be decayed 
    because doing so limits the model's ability to shift and scale its activations properly.'''
    decay_parameters = []
    no_decay_parameters = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    
    parameter_groups = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
        eps=1e-8,
    )

# Function to save model checkpoint
def save_checkpoint(
    model,
    optimizer,
    scaler,
    optimizer_step,
    tokens_seen,
    path,
    scheduler=None,
    history=None,
    best_val_loss=math.inf
):
    '''Training can take a lot of time, so to prepare for the crash, this function saves model weights, the optimizer state, the scaler state,
    the scheduler step, and the entire history of losses.'''
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
        "history": history,
        "best_val_loss": best_val_loss,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path) 

# Warmup function
def create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
        min_lr_ratio = 0.1,
):
    '''Models cannot handle a high learning rate on step 1. They output garbage, resulting in exploding gradients. This scheduler
    starts the learning rate near zero and linearly increases it over warmup_steps, allowing model to stabilize its initial random 
    weights. Once it hits maximum learning rate, it smoothly arcs downward following a cosine curve.

    '''
    if total_steps <= 0:
        raise ValueError("total steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup steps cannot be negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    
    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        
        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine 

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)

# Function that loads the model at specifed checkpoint
def load_checkpoint(
    model,
    optimizer,
    scaler,
    path,
    device,
    scheduler=None
):
    '''This function injects all the saved data back into RAM. The model continues training exactly as if the crash never happened. This function prevents amnesia 
    regarding previous learning. '''
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint

# Trains the model and return metrics recorded at each evaluation
def train_model(
        model, # Model 
        train_loader, # Data used to train the model
        val_loader, # Unseen data to validate learning
        optimizer, # Optimizer that adapts the learning rate
        device, # Device to run the training on preferably, cuda
        max_optimizer_steps, # Number of optimization steps to run the model for
        gradient_accumulation_steps=16, # Run the model this many times instead of a large batch
        eval_frequency=250, # Every 250 steps evaluate the model
        checkpoint_frequency=1_000, # Every 1000 steps save the model
        eval_batches = 50, # Evaluation batches
        scheduler = None, # Scheduler controls the learning rate
        checkpoint_dir = "checkpoints", # Checkpoint directory
        resume_from = None, # Start from a point of model crashes
        keep_step_checkpoints=False,
):
    '''The script trains the model through smaller batches and calculates the loss. If your batch size is 256, but
    the GPU cannot fit it all, the training will crash. This function loops through the micro steps (16) calculates loss, 
    accumualtes the loss and when micro_step % grad steps is 0, we update '''
    if max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if eval_frequency <= 0 or checkpoint_frequency <= 0:
        raise ValueError("Evaluation and checkpoint frequencies must be positive")
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty")

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler( # Because we are using Fp16, we get very small gradients, so GradScaler multiplies
        "cuda", # the loss by a large number before backpropogation, pushing gradients into a safe numerical range
        enabled=amp_enabled,
    )

    # History dict to track loss, learning rate, and time. Data is crucial for graphing
    history = {
        "optimizer_step": [],
        "tokens_seen": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "elapsed_seconds": [],
    }
    optimizer_step = 0
    micro_step = 0
    tokens_seen = 0
    best_val_loss = math.inf

    # If crash occured load the model from the checkpoint. 
    if resume_from is not None:
        checkpoint = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            path=resume_from,
            device=device,
            scheduler=scheduler,
        )
        optimizer_step = int(checkpoint.get("optimizer_step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        saved_history = checkpoint.get("history")
        if saved_history:
            history = saved_history

    optimizer.zero_grad(set_to_none=True) # Remove gradient tensors from memory to reduce memory consumption instead of zero.
    model.train() # Ensures dropout is actively dropping neurons

    accumulated_micro_loss = 0.0
    interval_loss_sum = 0.0
    interval_step_count = 0
    last_eval_step = history["optimizer_step"][-1] if history["optimizer_step"] else -1
    elapsed_offset = (
        float(history["elapsed_seconds"][-1])
        if history["elapsed_seconds"]
        else 0.0
    )

    start_time = time.perf_counter()
    # Print the progress in jupyter notebook
    progress = tqdm(
        total=max_optimizer_steps,
        initial=optimizer_step,
        desc="Training",
        unit="step",
    )

    # Function to evaluate the model at multiple distinct points. 
    def record_evaluation():
        '''This function calculates average training loss, pauses training and evaluates model to test the network on unseen data. It also logs the metrics like
        tokens seen, learning, rate and adds losses to the history dictionary, printing them to the terminal. If new validation loss is strictly lower,
        it overwrites the best.pt file to snure the highest performing version of the model is saved even if the model overfits and degrades later.'''
        nonlocal interval_loss_sum, interval_step_count, best_val_loss, last_eval_step

        train_loss = interval_loss_sum / max(1, interval_step_count)
        val_loss = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            max_batches=eval_batches
        )

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = elapsed_offset + (time.perf_counter() - start_time)
        
        history["optimizer_step"].append(optimizer_step)
        history["tokens_seen"].append(tokens_seen)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        history["elapsed_seconds"].append(elapsed)

        progress.set_postfix(
            train=f"{train_loss:.3f}",
            val=f"{val_loss:.3f}",
            lr=f"{current_lr:.2e}"
        )

        tqdm.write(
            f"Step {optimizer_step:,} | tokens {tokens_seen:,} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                optimizer_step=optimizer_step,
                tokens_seen=tokens_seen,
                path=checkpoint_dir/"best.pt",
                scheduler=scheduler,
                history=history,
                best_val_loss=best_val_loss
            )
        
        interval_loss_sum = 0.0
        interval_step_count = 0
        last_eval_step = optimizer_step
        
        return val_loss 
    
    try:
        # LLms are usually trained for a specific number of optimizer steps rather than epochs.
        while optimizer_step < max_optimizer_steps:
            for input_batch, target_batch in train_loader:
                input_batch = input_batch.to(device, non_blocking=True) # non_blocking allows the CPU to keep moving instead of waiting for GPU
                target_batch = target_batch.to(device, non_blocking=True)

                with _autocast_context(device):
                    logits=model(input_batch)
                    loss=F.cross_entropy(
                        logits.flatten(0,1),
                        target_batch.flatten(),
                    )
                    loss_for_backward = loss / gradient_accumulation_steps # Dividing the loss so that the final accumulated gradient matches the scale of a single massive batch
                
                scaler.scale(loss_for_backward).backward()

                accumulated_micro_loss += loss.detach().item()
                tokens_seen += input_batch.numel()
                micro_step += 1

                if micro_step % gradient_accumulation_steps != 0: # If the target hasn't been met continue training
                    continue 
                    
                scaler.unscale_(optimizer) # Reverses the artificial scaling applied earlier so the gradietns are in their true magnitude
                
                # Sometimes, bad batches cause massive loss spikes, if magnitue > 1.0 scale all gradients down proportionately
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer) # Updates the NN's weights based on the gradients if it detects infinite gradients, skips the update to protect the model
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # wipe the accumulated gradients from memory so the next batch starts clean

                if scheduler is not None:
                    scheduler.step() # tweak the learning rate following the warmup, cosine curve
                
                optimizer_step += 1
                step_train_loss = (
                    accumulated_micro_loss / gradient_accumulation_steps
                )
                accumulated_micro_loss = 0.0
                interval_loss_sum += step_train_loss
                interval_step_count += 1

                progress.update(1)
                progress.set_postfix(loss=f"{step_train_loss:.3f}")

                # If frequency reaches desired point record evaluation and save the model
                if optimizer_step % eval_frequency == 0:
                    record_evaluation()
                
                if optimizer_step % checkpoint_frequency == 0:
                    if keep_step_checkpoints:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            optimizer_step=optimizer_step,
                            tokens_seen=tokens_seen,
                            path=checkpoint_dir / f"step_{optimizer_step}.pt",
                            scheduler=scheduler,
                            history=history,
                            best_val_loss=best_val_loss
                        ) 
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        optimizer_step=optimizer_step,
                        tokens_seen=tokens_seen,
                        path=checkpoint_dir / "latest.pt",
                        scheduler=scheduler,
                        history=history,
                        best_val_loss=best_val_loss,
                    )
                
                if optimizer_step >= max_optimizer_steps:
                    break
            
                    
        if optimizer_step != last_eval_step:
            record_evaluation()
        

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            optimizer_step=optimizer_step,
            tokens_seen=tokens_seen,
            path=checkpoint_dir / "latest.pt",
            scheduler=scheduler,
            history=history,
            best_val_loss=best_val_loss,
        )
    finally:
        progress.close()

    return history

