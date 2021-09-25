from trainers.unet_trainer import UnetTrainer
import os


def main():

    unet_trainer = UnetTrainer("hyperparameters.yaml")
    unet_trainer.validate()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        exit()