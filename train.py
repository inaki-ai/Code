from trainers.unet_trainer import UnetTrainer


def main():

    unet_trainer = UnetTrainer("hyperparameters.yaml")
    unet_trainer.train()


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        exit()