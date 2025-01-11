import os
import torch
from pathlib import Path
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import CarvanaDataset, BasicDataset
from utils.dice_score import dice_loss
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    model = UNet(n_channels=3, n_classes=2)
    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)

    ckpt = './checkpoints/result.pth'
    state_dict = torch.load(ckpt, map_location=device)
    state_dict = {k: v for k,v in state_dict.items() if 'outc.conv' not in k}
    model_dict = model.state_dict()
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    dir_img = Path('./data/origin_new/')
    dir_mask = Path('./data/mask_new/')
    dir_checkpoint = Path('./checkpoints/')

    try:
        dataset = CarvanaDataset(dir_img, dir_mask, 0.5)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, 0.5)

    n_val = int(len(dataset) * 0.33)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-8)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.9)  # CHANGE
    grad_scaler = torch.amp.GradScaler('cuda', enabled=False)
    criterion = nn.CrossEntropyLoss()

    epochs = 1000
    best_val_score = float('-inf')
    best_loss = float('inf')
    last_loss = float('inf')
    early_stop = 0
    patience = 4

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, 'error1'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks_np = true_masks.cpu().numpy()
                true_masks_2d = true_masks_np.reshape(-1, true_masks_np.shape[-1])

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=False):
                    masks_pred = model(images)
                    assert true_masks.max().item() < model.n_classes, "标签值超出范围"
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])

                epoch_loss += loss.item()

            print(f'\nloss: {epoch_loss}\n')
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / f'LossResult.pth'))
                early_stop = 0
            elif epoch_loss > last_loss:
                early_stop += 1
                if early_stop > patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                # Evaluation round
            else:
                early_stop = 0
            last_loss = epoch_loss
                # Evaluation round
            if epoch % 2 == 0:
                val_score = evaluate(model, val_loader, device, False)
                scheduler.step(val_score)

                print(f'\nValidation Dice score: {val_score}\n')

                if val_score > best_val_score:
                    best_val_score = val_score
                    state_dict = model.state_dict()
                    state_dict['mask_values'] = dataset.mask_values
                    torch.save(state_dict, str(dir_checkpoint / f'ValResult.pth'))

