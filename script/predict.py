import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report, accuracy_score

class Predict():

    def __init__(self, file:str, iftest:bool) -> None:
        self.file = file
        self.iftest = iftest
        
    def run(self):
        feature = np.loadtxt(f"./data/{self.file}.txt")
        result = np.zeros((len(feature)))
        ypred = np.zeros((len(feature)))
        df = pd.DataFrame(columns=['model1','model2','model3','model4','model5'])
        for i in range(5):
            model = joblib.load("./model/" + str(i + 1) + ".model")
            #result = result + model.predict(feature)
            result = model.predict_proba(feature)[:, 1]
            df[f'model{i+1}'] = result
            ypred += result/5
            #result = model.predict(feature)
            # print("--------------------model " + str(i) + "------------------------")
            # for i in result:
            #     print(i)
        df.to_excel(f'./result/{self.file}.xlsx', sheet_name=self.file.split('/')[-1], index=False)
        print(f'save results in result/{self.file}.xlsx')
        print('Done.')
        if self.iftest:
            yreal = np.loadtxt(f"./data/y_{self.file}.txt")
            for i in range(len(ypred)):
                if ypred[i]>0.5:
                    ypred[i]=1
                else:
                    ypred[i]=0
            print(yreal)
            print(ypred)
            cm = confusion_matrix(yreal, ypred)
            print(f'precision: {(cm[0,0]+cm[1,1])/np.sum(cm)}')
            print(f'recall of 1: {cm[1,1]/np.sum(cm[1])}')
            print(f'precision of 1: {cm[1,1]/np.sum(cm[:,1])}')

    def run_test(self):
        feature = np.loadtxt(f"./data/{self.file}.txt")
        result = np.zeros((len(feature)))
        ypred = np.zeros((len(feature)))
        df = pd.DataFrame(columns=['model1','model2','model3','model4','model5'])
        yreal = np.loadtxt(f"./data/y_{self.file}.txt")
        for i in range(5):
            print("--------------------model " + str(i) + "------------------------")
            model = joblib.load("./model/" + str(i + 1) + ".model")
            #result = result + model.predict(feature)
            result = model.predict_proba(feature)[:, 1]
            df[f'model{i+1}'] = result
            ypred += result/5
            for i in range(len(result)):
                if result[i]>0.5:
                    result[i]=1
                else:
                    result[i]=0
            print(yreal)
            print(result)
            cm = confusion_matrix(yreal, result)
            print(f'precision: {(cm[0,0]+cm[1,1])/np.sum(cm)}')
            print(f'recall of 1: {cm[1,1]/np.sum(cm[1])}')
            print(f'precision of 1: {cm[1,1]/np.sum(cm[:,1])}')

if __name__ == '__main__':
    test = Predict('Data', False)
    test.run()
