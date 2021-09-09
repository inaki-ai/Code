from trainers.unet_trainer import UnetTrainer


def main():

    unet_trainer = UnetTrainer("hyperparameters.yaml")


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        exit()