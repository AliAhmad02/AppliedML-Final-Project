import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_epoch(model, optimizer, loss_fn, train_loader, device, scaler):
    model.train()
    total_loss = 0.0

    for batch_img, batch_num, batch_target in train_loader:
        batch_img, batch_num, batch_target = (
            batch_img.to(device),
            batch_num.to(device),
            batch_target.to(device),
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast():  # Use autocast to automatically handle mixed precision
            predictions = model(batch_img, batch_num)
            loss = loss_fn(predictions, batch_target)

        scaler.scale(loss).backward()  # Scale the loss value
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, loss_fn, test_loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_img_test, batch_num_test, batch_target_test in test_loader:
            batch_img_test, batch_num_test, batch_target_test = (
                batch_img_test.to(device),
                batch_num_test.to(device),
                batch_target_test.to(device),
            )
            predictions_test = model(batch_img_test, batch_num_test)
            loss = loss_fn(predictions_test, batch_target_test)

            total_loss += loss.item()

    return total_loss / len(test_loader)


def train_model(model, optimizer, loss_fn, train_loader, test_loader, device, n_epochs, scheduler=None):
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training
    train_hist = []
    test_hist = []

    for epoch in tqdm(range(n_epochs)):
        train_loss = train_epoch(
            model, optimizer, loss_fn, train_loader, device, scaler)
        test_loss = validate(model, loss_fn, test_loader, device)

        train_hist.append(train_loss)
        test_hist.append(test_loss)

        if scheduler:
            scheduler.step()  # Step the scheduler at the end of each epoch

        print(
            f"Epoch [{epoch+1}/{n_epochs}] - Training Loss: {
                train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    return train_hist, test_hist
