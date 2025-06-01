from GA_structures import Year, GeneticAlgorithm

def main():

    path = 'Dane/Year1'

    evalFunctionCoeffs = {
        "teacher" : 10,
        "classroom" : 10,
        "subjectHoursMissmatch" : 10
    }

    ga = GeneticAlgorithm(path, 100, 7, evalFunctionCoeffs, GeneticAlgorithm.SelectionModel.ELITE)

    ga.run()

if __name__ == "__main__":
    main()