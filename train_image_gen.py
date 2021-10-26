from trainers.new_data_generator_trainer import DataGenerator
import os
import yaml


def main():

    file = open("hyperparameters/hyperparameters.yaml", 'r')
    parameter_dict = yaml.safe_load(file)

    if parameter_dict['mlflow']:

        import mlflow

        mlflow.set_experiment(parameter_dict["net"] + " WGAN")

        with mlflow.start_run():
            data_generator = DataGenerator("hyperparameters.yaml")
            data_generator.train()

    else:
        data_generator = DataGenerator("hyperparameters.yaml")
        data_generator.train()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        exit()