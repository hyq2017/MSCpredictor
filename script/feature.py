import pandas as pd
import numpy as np
import gensim.models as gm

class featureExtract:
    def __init__(self, file:str) -> None:
        self.file = file
        print(f'read input from data/{self.file}.csv...')
        self.data = pd.read_csv(f'./data/{self.file}.csv', sep='\t')
        # self.__properties_dict = {
        #     'A': np.array([89.09, 0, 0, 0, 1, 0, 0]),
        #     'C': np.array([121.16, 1, 0, 0, 1, 0, 0]),
        #     'D': np.array([133.10, 1, -1, 1, -1, 0, 1]),
        #     'E': np.array([147.13, 1, -1, 1, -1, 0, 1]),
        #     'F': np.array([165.19, 0, 0, 0, 1, 0, 0]),
        #     'G': np.array([75.07, 1, 0, 0, 0, 0, 0]),
        #     'H': np.array([155.16, 1, 1, -1, 0, 1, 1]),
        #     'I': np.array([131.17, 0, 0, 0, 1, 0, 0]),
        #     'K': np.array([146.19, 1, 1, -1, -1, 1, 0]),
        #     'L': np.array([131.17, 0, 0, 0, 1, 0, 0]),
        #     'M': np.array([149.21, 0, 0, 0, 1, 0, 0]),
        #     'N': np.array([132.12, 1, 0, 0, -1, 1, 1]),
        #     'P': np.array([115.13, 0, 0, 0, 0, 0, 0]),
        #     'Q': np.array([146.15, 1, 0, 0, -1, 1, 1]),
        #     'R': np.array([174.20, 1, 1, -1, -1, 1, 0]),
        #     'S': np.array([105.09, 1, 0, 0, 0, 1, 1]),
        #     'T': np.array([119.16, 1, 0, 0, 0, 1, 1]),
        #     'V': np.array([117.15, 0, 0, 0, 1, 0, 0]),
        #     'W': np.array([204.22, 0, 0, 0, 1, 1, 0]),
        #     'Y': np.array([181.19, 1, 0, 0, 0, 1, 1]),
        #     'X': np.array([136.90, 1, 0, 0, 1, 0, 0]),
        # }
        temp = pd.read_csv('./model/AAFeature/aaindex_feature.txt', sep='\t').iloc[:, list(np.array([-1, 452, 217, 92, 405, 48, 15, 330])+1)] # 7soft

        self.__properties_dict = temp.set_index('AA').T.to_dict('list')
        self.__properties_dict['X'] = temp.mean(axis=0, numeric_only=True).tolist()
        # input(self.__properties_dict)

    def __position_feature(self):
        feature_pos = np.array(())
        position = self.data['Mutation Resi']
        sequence = self.data['Uniprot_Sequence']
        for i in range(len(position)):
            feature_windows_1 = np.array(())
            feature_windows_2 = np.array(())
            feature_windows_3 = np.array(())
            feature_windows_4 = np.array(())
            # 4 is the window size
            for j in [2, 3, 4, 5, 6]:
                # mutation position is (position[i] - 1)
                if position[i] - j < 0:
                    feature = np.array(self.__properties_dict['X'])
                else:
                    feature = np.array(self.__properties_dict[str(sequence[i][position[i] - j])])

                #feature = feature[:, np.newaxis]
                feature_windows_1 = np.hstack((feature_windows_1, feature))
            feature_windows_1 = feature_windows_1.reshape(-1, 7) # 7 is the dimension of properties
            feature_1 = np.max(feature_windows_1, axis=0) - np.min(feature_windows_1, axis=0)

            for j in [7, 8, 9, 10, 11]:
                if position[i] - j < 0:
                    feature = np.array(self.__properties_dict['X'])
                else:
                    feature = np.array(self.__properties_dict[str(sequence[i][position[i] - j])])
                feature_windows_2 = np.hstack((feature_windows_2, feature))
            feature_windows_2 = feature_windows_2.reshape(-1, 7)
            feature_2 = np.max(feature_windows_2, axis=0) - np.min(feature_windows_2, axis=0)

            for j in [0, 1, 2, 3, 4]:
                if int(position[i]) + j >= len(sequence[i]):
                    feature = np.array(self.__properties_dict['X'])
                else:
                    feature = np.array(self.__properties_dict[str(sequence[i][position[i] + j])])
                feature_windows_3 = np.hstack((feature_windows_3, feature))
            feature_windows_3 = feature_windows_3.reshape(-1, 7) # 7 is the dimension of properties
            feature_3 = np.max(feature_windows_3, axis=0) - np.min(feature_windows_3, axis=0)

            for j in [5, 6, 7, 8, 9]:
                if position[i] + j >= len(sequence[i]):
                    feature = np.array(self.__properties_dict['X'])
                else:
                    feature = np.array(self.__properties_dict[str(sequence[i][position[i] + j])])
                feature_windows_4 = np.hstack((feature_windows_4, feature))
            feature_windows_4 = feature_windows_4.reshape(-1, 7)  # 7 is the dimension of properties
            feature_4 = np.max(feature_windows_4, axis=0) - np.min(feature_windows_4, axis=0)


            feature_final = np.concatenate((feature_1, feature_2, feature_3, feature_4))
            feature_pos = np.hstack((feature_pos, feature_final))
        feature_pos = feature_pos.reshape(-1,28)
        return feature_pos

    def __mutation_feature(self):
        mutation = self.data['Mutation Type']
        feature = np.array(())
        for i in range(len(mutation)):
            mutation_old = mutation[i][0]
            mutation_new = mutation[i][3]
            feature_old = np.array(self.__properties_dict[str(mutation_old)])
            feature_new = np.array(self.__properties_dict[str(mutation_new)])
            feature_difference = feature_new - feature_old
            feature_mean = (feature_old + feature_new) / 2
            feature = np.hstack((feature, feature_difference))
            # feature = np.hstack((feature, feature_difference))
            # feature = np.hstack((feature, feature_mean))
        feature = feature.reshape(-1,7)
        # feature = feature.reshape(-1,14)
        return feature

    def __sequence_feature(self):
        sequence = self.data['Uniprot_Sequence']
        feature = np.array(())
        dict_w2v = gm.KeyedVectors.load(
            ".\model\pretrained\w2v_3mer_50dim.vector",
            mmap='r'
        )
        for seq in sequence:
            kmer = [seq[i:i+3] for i in range(len(seq)-3)]
            matrix = []
            for i in kmer:
                matrix.append(dict_w2v[i])
            feature = np.hstack((feature, sum(matrix) / len(matrix)))
        feature = feature.reshape(-1,50)
        return feature

    def run(self):
        print('特征提取...')
        positional_feature = self.__position_feature()
        mutational_feature = self.__mutation_feature()
        sequential_feature = self.__sequence_feature()
        feature_final = np.hstack((positional_feature, mutational_feature, sequential_feature))
        np.savetxt(f"./data/{self.file}.txt", feature_final)
        print(f'save features in data/{self.file}.txt')
        print('Done.')

if __name__ == '__main__':
    test = featureExtract('0615')
    test.run()
