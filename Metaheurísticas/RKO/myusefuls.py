'''
Métodos de utilidades
'''
import numpy as np 

class Useful:
    '''
    Classe para trabalhar com os arquivos das instâncias.
    '''

    @staticmethod
    def read_data(path: str, file: str):
        filename = path  + file
        with open(filename, "r") as f:
            
            data = f.read()
            instances = data.split("\n")

            strs = [linha.replace(" ", "\t") for linha in instances]        
                
            nPontos, nHubs, alphaH = int(strs[0].split('\t')[0]), int(strs[0].split('\t')[1]), float(strs[0].split('\t')[2])

            N  = nPontos + (nPontos - nHubs) + (nHubs * (nHubs - 1)/2)

            dist, flows = np.zeros((nPontos, nPontos)), np.zeros((nPontos, nPontos))

            for line in strs[1:]:
                values = line.strip().split()

                i = int(values[0])
                j = int(values[1])
                f = float(values[2])

                if file[6:9] == 'CAB':
                    d = float(values[3]) / 1000000
                else: 
                    d = float(values[3]) / 1000

                dist[i][j] = d
                flows[i][j] = f

            return nPontos, nHubs, alphaH, N, dist, flows
        
    @staticmethod
    def RunScenario()-> list:
        '''
        Função que lê o arquivo testScenario, o qual contém o nome da instância, a quantidade de runs de cada MH e o valor ótimo para instância
        '''

        path_instances = "//Users/arthurdantasmangussi/@Mestrado ITA/PO-205/Instâncias//"
        filename = path_instances + 'testScenario.txt'

        testScenario = []
        with open(filename, "r") as file:
            data = file.read()
            scenarios = data.split("\n")
            for line in scenarios[1:]:
                values = line.strip().split()

                name = str(values[0])
                run = int(values[1])
                optimal = float(values[2])

                testScenario.append([name, run, optimal])
            return testScenario
        
    @staticmethod
    def SaveResults(MHresults:list, name_file_output:str):
        with open(f'{name_file_output}.txt', 'w') as f:
            for result in MHresults:
                f.write(f'{result[0]}     {result[1]}     {result[2]}     {result[3]}' + '\n')

    @staticmethod 
    def ParametersMH():
        """
        Função que armazena os parâmetros de cada metaheurística
        """
        return
    
    

            
        

            

        
        

            