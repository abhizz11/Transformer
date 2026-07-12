import torch
import torch.nn.functional as F
from pathlib import Path


def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device, 
        max_optimizer_steps,
        gradient_accumulation_steps=16,
        eval_frequency=250,
        checkpoint_frequency=1_000,
):
    amp_enabled = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    optimizer.zero_grad(set_to_none=True)

    optimizer_step = 0
    micro_step = 0
    tokens_seen = 0

    model.train()

    while optimizer_step < max_optimizer_steps:
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(
                device,
                non_blocking=True
            )
            target_batch = target_batch.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(input_batch)

                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    target_batch.flatten(),
                )

                scaled_loss = (
                    loss / gradient_accumulation_steps
                )
            
            scaler.scale(scaled_loss).backward()

            tokens_seen += input_batch.numel()
            micro_step += 1

            if micro_step % gradient_accumulation_steps != 0:
                continue 

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1

            print(
                f"Step {optimizer_step:06d} | "
                f"loss {loss.item():.4f} | "
                f"tokens {tokens_seen:,}"
            )

            if optimizer_step % eval_frequency == 0:
                # Call your existing validation function here.
                pass 

            if optimizer_step % checkpoint_frequency == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    optimizer_step=optimizer_step,
                    tokens_seen=tokens_seen,
                    path=f"checkpoints/step_{optimizer_step}.pt",
                )

            if optimizer_step >= max_optimizer_steps:
                break





def save_checkpoint(
    model,
    optimizer,
    scaler,
    optimizer_step,
    tokens_seen,
    path,
):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "optimizer_step": optimizer_step,
            "tokens_seen": tokens_seen,
        },
        checkpoint_path,
    )

def load_checkpoint(
    model,
    optimizer,
    scaler,
    path,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    return (
        checkpoint["optimizer_step"],
        checkpoint["tokens_seen"],
    )