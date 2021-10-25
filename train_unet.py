from trainers.unet_trainer import UnetTrainer
import os
import mlflow
import yaml


def main():

    file = open("hyperparameters/hyperparameters.yaml", 'r')
    parameter_dict = yaml.safe_load(file)

    mlflow.set_experiment(parameter_dict["net"])

    with mlflow.start_run():
        unet_trainer = UnetTrainer("hyperparameters.yaml")
        unet_trainer.train()
        #unet_trainer.generate_adversarial_examples()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
        mlflow.end_run()
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        mlflow.end_run()
        exit()