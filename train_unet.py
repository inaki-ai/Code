from trainers.unet_trainer import UnetTrainer
import os
import yaml


def main():
    
    unet_trainer = UnetTrainer("hyperparameters.yaml")
    unet_trainer.train()
    unet_trainer.test()
    #unet_trainer.generate_augmented_examples()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        exit()