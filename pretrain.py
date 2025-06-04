import logging
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from pretraining.datasets.utils import TactilePlayDataset
from pretraining.models.build_model import build_model
from pretraining.pretrain_utils import build_args, TBLogger, create_optimizer, get_current_lr, set_seed
from robomimic.models.utils import data_to_gnn_batch

from diffusion_policy.real_world.constants import TACTILE_RAW_DATA_SCALE

dataset_path = 'data/expert_dataset_holodex/'
eval_every = 100

def prediction_evaluation(model, test_dataloader, local_epoch_idx, logger, edge_type, device):
    model.eval()
    total_mask_data_force_error = []
    total_mask_data_zero_force_error = []
    total_mask_data_nonzero_force_error = []
    total_keep_data_force_error = []
    total_all_data_force_error = []
    total_resultant_force_error = []
    with torch.no_grad():
        with tqdm(test_dataloader, desc=f"Testing epoch {local_epoch_idx}", 
                    leave=False, mininterval=1) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch['tactile_data'] = batch['tactile_data'].to(device, torch.float32)
                    if model.resultant_type == 'force':
                        batch['resultant_data'] = batch['resultant_data'].to(device, torch.float32)
                    batch['tactile_data'], _, _, _ = data_to_gnn_batch(batch['tactile_data'], edge_type)
                    x, recon, mask_nodes, keep_nodes, predicted_resultant_force = model(batch, return_predict=True, eval=True)
                    
                    # pairwise distance
                    tactile_scale = torch.tensor(TACTILE_RAW_DATA_SCALE, device=x.device)
                    if model._mask_rate > 0:
                        assert model.mask_index == 9, "following error calculation is only for mask_index 9, e.g, 9-12 is force"
                        mask_data_force_error = torch.sqrt(torch.nn.functional.mse_loss(x[mask_nodes][:,-3:]*tactile_scale, recon[mask_nodes][:,-3:]*tactile_scale))
                        keep_data_force_error = torch.sqrt(torch.nn.functional.mse_loss(x[keep_nodes][:,-3:]*tactile_scale, recon[keep_nodes][:,-3:]*tactile_scale))
                        all_data_force_error = torch.sqrt(torch.nn.functional.mse_loss(x[:,-3:]*tactile_scale, recon[:,-3:]*tactile_scale))

                        original_zero_ids = (torch.sum(x[mask_nodes][:,-3:],-1)==0).nonzero(as_tuple=False).squeeze(-1)
                        original_nonzero_ids = (torch.sum(x[mask_nodes][:,-3:],-1)!=0).nonzero(as_tuple=False).squeeze(-1)
                        mask_data_zero_force_error = torch.sqrt(torch.nn.functional.mse_loss(x[mask_nodes][original_zero_ids,-3:]*tactile_scale, recon[mask_nodes][original_zero_ids,-3:]*tactile_scale))
                        mask_data_nonzero_force_error = torch.sqrt(torch.nn.functional.mse_loss(x[mask_nodes][original_nonzero_ids,-3:]*tactile_scale, recon[mask_nodes][original_nonzero_ids,-3:]*tactile_scale))

                    if model.resultant_type == 'force':
                        resultant_force_error = torch.sqrt(torch.nn.functional.mse_loss(batch['resultant_data']*tactile_scale, predicted_resultant_force*tactile_scale))
                        total_resultant_force_error.append(resultant_force_error.item())
                    if model._mask_rate > 0:
                        total_mask_data_force_error.append(mask_data_force_error.item())
                        total_mask_data_zero_force_error.append(mask_data_zero_force_error.item())
                        total_mask_data_nonzero_force_error.append(mask_data_nonzero_force_error.item())
                        total_keep_data_force_error.append(keep_data_force_error.item())
                        total_all_data_force_error.append(all_data_force_error.item())

    eval_dict = {}
    if model._mask_rate > 0:
        eval_dict["eval/mask_data_force_error (unit)"] = np.mean(total_mask_data_force_error)
        eval_dict["eval/mask_data_zero_force_error (unit)"] = np.mean(total_mask_data_zero_force_error)
        eval_dict["eval/mask_data_nonzero_force_error (unit)"] = np.mean(total_mask_data_nonzero_force_error)
        eval_dict["eval/keep_data_force_error (unit)"] = np.mean(total_keep_data_force_error)
        eval_dict["eval/all_data_force_error (unit)"] = np.mean(total_all_data_force_error)
    if model.resultant_type == 'force':
        eval_dict["eval/resultant_force_error (unit)"] = np.mean(total_resultant_force_error)

    print(eval_dict)
    # if model._mask_rate > 0:
    #     print('------------------')
    #     print(x[mask_nodes[0]])
    #     print('==================')
    #     print(recon[mask_nodes[0]])
    logger.note(eval_dict, step=local_epoch_idx)

def pretrain(model, optimizer, train_dataloader, test_dataloader, max_epoch, device, scheduler, logger=None, edge_type='four+sensor'):
    logging.info("start training..")
    
    for local_epoch_idx in range(max_epoch):
        # ========= train for this epoch ==========
        # create loss buffer
        train_losses = {}
        train_losses['recon_loss'] = []
        if model.resultant_type == 'force':
            train_losses['predict_loss'] = []
        
        # train the model
        with tqdm(train_dataloader, desc=f"Training epoch {local_epoch_idx}", 
                leave=False, mininterval=1) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                model.train()
                batch['tactile_data'] = batch['tactile_data'].to(device, torch.float32)
                if model.resultant_type == 'force':
                    batch['resultant_data'] = batch['resultant_data'].to(device, torch.float32)
                # convert to gnn batch
                batch['tactile_data'], _, _, _ = data_to_gnn_batch(batch['tactile_data'], edge_type)

                # forward pass
                optimizer.zero_grad()
                loss, loss_dict = model(batch)

                
                train_losses['recon_loss'].append(loss_dict['recon_loss'])
                if model.resultant_type == 'force':
                    train_losses['predict_loss'].append(loss_dict['predict_loss'])
                
                loss.backward()
                optimizer.step()
                
                # scheduler step
                if scheduler is not None and (batch_idx+1)%int(len(train_dataloader)/2)==0 and get_current_lr(optimizer)>1e-5:
                    scheduler.step()

                tepoch.set_description(f"# Epoch {local_epoch_idx}: train_loss: {loss.item():.4f}")

            # log the loss
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                loss_dict["recon_loss"] = np.mean(train_losses['recon_loss'])
                if model.resultant_type == 'force':
                    loss_dict["predict_loss"] = np.mean(train_losses['predict_loss'])
                logger.note(loss_dict, step=local_epoch_idx)

            # evaluate the model every eval_every epochs
            if (local_epoch_idx) % eval_every == 0:
                prediction_evaluation(model, test_dataloader, local_epoch_idx, logger, edge_type, device)
            
            # save the model every 500 epochs
            if (local_epoch_idx) % 500 == 0:
                torch.save(model.encoder.nets.state_dict(), os.path.join(logger.save_path, f"checkpoint_{local_epoch_idx}.pt"))
                torch.save(model.state_dict(), os.path.join(logger.save_path, f"checkpoint_whole_{local_epoch_idx}.pt"))

    # return best_model
    return model

def main(args):
    # process args
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    set_seed(seeds[0])
    train_dataset_name = args.train_dataset
    test_dataset_name = args.test_dataset
    max_epoch = args.max_epoch
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 

    lr = args.lr
    weight_decay = args.weight_decay
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    eval_ckpt_path = args.eval_ckpt_path

    # build datasets and dataloaders, the process of converting raw tactile into canonical representation is done in the dataset
    train_dataset = TactilePlayDataset(os.path.join(dataset_path, train_dataset_name), device, resultant_type=args.resultant_type, aug_type=args.aug_type, valid=False)
    test_dataset = TactilePlayDataset(os.path.join(dataset_path, test_dataset_name), device, resultant_type=args.resultant_type, aug_type=args.aug_type, valid=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # override args with dataset info
    args.num_features = train_dataset.get_info()['num_features']
    args.num_nodes = train_dataset.get_info()['num_nodes']
    print(args.num_features, args.edge_type=='four+sensor')

    # set up logging
    if logs:
        logger = TBLogger(name=f"{args.exp_name}_{args.edge_type}_train:{train_dataset_name}_test:{test_dataset_name}_rpr_{replace_rate}__mp_{max_epoch}_wd_{weight_decay}_{encoder_type}_{decoder_type}")
    else:
        logger = None

    # build model
    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    # setup scheduler
    if use_scheduler:
        logging.info("Use schedular")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None
    
    # start training
    if not load_model and eval_ckpt_path is None:
        model = pretrain(model, optimizer, train_loader, test_loader, max_epoch, device, scheduler, logger, args.edge_type)
        model = model.cpu()

    # done training, now evaluate
    if load_model or eval_ckpt_path is not None:
        logging.info("Loading Model ... ")
        model.load_state_dict(torch.load(eval_ckpt_path))
        model.to(device)
        prediction_evaluation(model, test_loader, 0, logger, args.edge_type, args.device)

    # save final encoder only
    if save_model:
        logging.info("Saveing Model ...")
        torch.save(model.encoder.nets.state_dict(), os.path.join(logger.save_path, "checkpoint.pt"))

    # finish logging
    if logger is not None:
        logger.finish()
   
if __name__ == "__main__":
    args = build_args()
    print(args)
    main(args)