import abc


class BaseDetector(abc.ABC):
    """
    A base class for detectors.
    The following should be overridden:
    load_model()
    preprocess()
    inference()
    """
    def __init__(self, width, height, thresh):
        self.width = width
        self.height = height
        self.thresh = thresh
        self.model = None

    @abc.abstractmethod
    def load_model(self, model_path):
        raise NotImplementedError

 
    @abc.abstractmethod
    def preprocess(self, raw_image):
        raise NotImplementedError

 
    @abc.abstractmethod
    def inference(self, preprocessed_image):
        raise NotImplementedError

  
