def train(model, criterion, optimizer, scheduler, dataloader, num_epochs, device, writer):
    model.to(device)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.double().to(device), labels.double().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if i == 0 and epoch == 0:
                print(f"First loss = {loss}")
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # Log the model parameters and gradients (optional)
        for name, param in model.named_parameters():
            writer.add_histogram(f'{name}/weights', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}/gradients', param.grad, epoch)

        if epoch % 25 == 0:
            print(f'Epoch: {epoch}, Loss: {avg_loss}, Learning Rate: {scheduler.get_last_lr()[0]}')

    print('Finished Training')
