from datetime import datetime

class Settings:
    RESOLUTION = 1.0
    DRAW = False
    K_TOPICS = 10
    PROJECT_NAME = ''
    DIRECTORY = 'V:\\papercode\\pythonProject'
    DATA_DIRECTORY = f'{DIRECTORY}\\data'
    PROJECT_PATH = ''
    STOP_WORDS_PATH = 'V:\\papercode\\pythonProject\\stop_words.txt'
    TOPIC_MODEL_PATH = 'V:\\papercode\\pythonProject\\topic_model_results.txt'


    @staticmethod
    def create_id():
        Settings.ID = datetime.now().strftime(
            f'%d_%m_%H_%M_%S')

