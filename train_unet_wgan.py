from trainers.wgan_trainer import WGanTrainer
import os
import mlflow
import yaml


def main():

    file = open("hyperparameters/hyperparameters.yaml", 'r')
    parameter_dict = yaml.safe_load(file)

    mlflow.set_experiment(parameter_dict["net"] + " WGAN")

    with mlflow.start_run():
        wgan_trainer = WGanTrainer("hyperparameters.yaml")
        wgan_trainer.train()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
        mlflow.end_run()
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        mlflow.end_run()
        exit()