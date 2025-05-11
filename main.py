from GA_structures import Year, GeneticAlgorithm

def main():

    path = 'Dane/Year1'

    ga = GeneticAlgorithm(path, 100, 15, GeneticAlgorithm.SelectionModel.ELITE)

    ga.run()

if __name__ == "__main__":
    main()