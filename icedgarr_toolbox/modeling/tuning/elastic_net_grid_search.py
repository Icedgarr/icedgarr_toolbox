class ElasticNetGridSearch:
    def sample_space(self):
        spaces = self.hyperparameter_spaces()
        return [{'alpha': alpha, 'l1_ratio': l1} for alpha in spaces['alpha'] for l1 in spaces['l1_ratio']]

    def hyperparameter_spaces(self):
        hyperparameter_spaces = {
            'alpha': [0.01, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 1],
            'l1_ratio': [.1, .5, .7, .9, .95, .99, 1]
        }
        return hyperparameter_spaces
