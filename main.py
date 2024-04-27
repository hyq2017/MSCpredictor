from script.feature import featureExtract
from script.predict import Predict


def main():
    fe = featureExtract('test')
    fe.run()
    pt = Predict('test', False)
    pt.run()


if __name__ == "__main__":
    main()

