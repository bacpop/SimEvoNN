## This script will include the classes for the alleles.
# simulated allele has a distinct positive or negative selection coefficient.

class Allele:
    def __init__(self, name, selection_coefficient):
        self.name = name
        self.selection_coefficient = selection_coefficient

    def calculate_fitness(self, frequency):
        return frequency * self.selection_coefficient

    def calculate_selection_coefficient(self):
        ##output depends on the "functional analysis"
        # https://www.nature.com/articles/s41559-017-0337-x
        ##for now, just return the selection coefficient
        return self.selection_coefficient

    def calculate_conjugation_chance(self):
        ...