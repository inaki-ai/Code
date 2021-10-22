from trainers.wgan_trainer import WGanTrainer
import os


def main():

    wgan_trainer = WGanTrainer("hyperparameters.yaml")
    wgan_trainer.test()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        exit()
