import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import dataloader
from utils import metrics
from backbone.res_unet import ResUnet
from backbone.res_unet_plus import ResUnetPlusPlus
from utils.logger import MyWriter
import torch
import argparse
import os


def main(hp, num_epochs, resume, load_from, name):

    checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, name))
    # get model

    model = ResUnetPlusPlus(3)
 
    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # starting params
    best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            # checkpoint = torch.load(resume)
            checkpoint = torch.load('/home/thanh/workspace/intern/resunetpp_custom/checkpoints/default/exp8_512.pt', map_location=torch.device('cpu'))
            start_epoch = checkpoint["epoch"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"], strict=True)
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if load_from:
        if os.path.isfile(load_from):
            print("=> loading checkpoint '{}'".format(load_from))
            # checkpoint = torch.load(load_from)
            # model.load_state_dict(checkpoint["state_dict"])
            checkpoint = torch.load(load_from, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    load_from, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.load_from))
    # get data
    mass_dataset_train = dataloader.ImageDataset(
        hp, transform=transforms.Compose([dataloader.ToTensorTarget()])
    )

    mass_dataset_val = dataloader.ImageDataset(
        hp, False, transform=transforms.Compose([dataloader.ToTensorTarget()])
    )

    # creating loaders
    train_dataloader = DataLoader(
        mass_dataset_train, batch_size=hp.batch_size, num_workers=2, shuffle=True
    )
    val_dataloader = DataLoader(
        mass_dataset_val, batch_size=1, num_workers=2, shuffle=False
    )

    step = 0
    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        # logging accuracy and loss
        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()
        # iterate over data

        loader = tqdm(train_dataloader, desc="training")
        for idx, data in enumerate(loader):

            # get the inputs and wrap in Variable
            inputs = data["sat_img"]
            labels = data["map_img"]
            labels = labels[..., 0]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # prob_map = model(inputs) # last activation was a sigmoid
            # outputs = (prob_map > 0.3).float()
            print(inputs.shape)
            outputs = model(inputs)
            # outputs = torch.nn.functional.sigmoid(outputs)

            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))

            # tensorboard logging
            if step % hp.logging_step == 0:
                writer.log_training(train_loss.avg, train_acc.avg, step)
                loader.set_description(
                    "Training Loss: {:.4f} Acc: {:.4f}".format(
                        train_loss.avg, train_acc.avg
                    )
                )

            # Validatiuon
            if step % hp.validation_interval == 0:
                valid_metrics = validation(
                    val_dataloader, model, criterion, writer, step
                )
                save_path = os.path.join(
                    checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, step)
                )
                # store best loss and save a model checkpoint
                best_loss = min(valid_metrics["valid_loss"], best_loss)
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "arch": "ResUnet",
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

            step += 1


def validation(valid_loader, model, criterion, logger, step):

    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc="validation")):

        # get the inputs and wrap in Variable
        inputs = data["sat_img"]
        labels = data["map_img"]
        labels = labels[..., 0]

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        # outputs = torch.nn.functional.sigmoid(outputs)

        loss = criterion(outputs, labels)

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))
        if idx == 0:
            logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)
    logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}


if __name__ == "__main__":
    # Create an object parser 
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    # add_argument 
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    # default = 75
    parser.add_argument(
        "--epochs",
        default= 10, 
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--load_from",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--name", default="default", type=str, help="Experiment name")

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())
    
    main(hp, num_epochs=args.epochs, resume=args.resume, load_from=args.load_from, name=args.name)
