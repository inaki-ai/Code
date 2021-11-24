from trainers.wgan_trainer import WGanTrainer
import os
import yaml


def main():

    wgan_trainer = WGanTrainer("hyperparameters-WGAN.yaml")
    wgan_trainer.train()
    wgan_trainer.test()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        exit()