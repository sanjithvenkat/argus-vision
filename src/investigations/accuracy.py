from .base import Investigation

class AccuracyInvestigation(Investigation):

    def run(self, model, dataset):
        print("Running Accuracy Investigation...")