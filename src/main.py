from image_preprocessor import GalactiClass_PreProcessing
from morphology_detector import GalactiClass_MorphologyDetector
from helpers import GalacticClass_Helpers
from common_imports import *

class Main:
    def __init__(self):
        self.morphology_detector = GalactiClass_MorphologyDetector()

    def run(self):
        # Main execution flow
        pass

if __name__ == "__main__":
    main = Main()
    main.run()