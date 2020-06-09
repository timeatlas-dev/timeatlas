from timeatlas import AnomalyGeneratorTemplate


def create_template(filename, seed, functions, threshold, num_animalies, anomaly_name):
    template = AnomalyGeneratorTemplate(filename=filename,
                                        seed=seed,
                                        functions=functions,
                                        threshold=threshold,
                                        num_anomalies=num_animalies,
                                        anomaly_name=anomaly_name)

    template.write()


if __name__ == '__main__':
    filename = "test"
    seed = 1234
    functions = ['flatline', 'hard_knee']
    threshold = 1234
    num_animalies = None
    write = True
    anomaly_name = "ANOMALY"

    create_template(filename, seed, functions, threshold, num_animalies, anomaly_name)
