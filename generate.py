from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np,tqdm
from visionlanguage import OCRConflictGenerator, FigureConflictGenerator, GeometricConflictGenerator, SemanticConflictGenerator
import openai,os,re
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
COLOR_CONFIG = yaml.load(open('colors.yaml'),Loader=yaml.FullLoader)
PROMPT_TEMPLATE = yaml.load(open('prompt.yaml'),Loader=yaml.FullLoader)
os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
openai.api_key = OPENAI_CONFIG['OPENAI_KEY']
try:
    openai.api_base = OPENAI_CONFIG['OPENAI_API_BASE']
except:
    pass
model = OPENAI_CONFIG['MODEL_NAME']
max_retry = OPENAI_CONFIG['MAX_RETRY']


fcg = FigureConflictGenerator(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry)
fcg.get_dict()
fcg.create()

# ocg = OCRConflictGenerator(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry)
# ocg.grow_sentence_list(num=5)
# ocg.create()

# gcg = GeometricConflictGenerator()
# gcg.create(num=5)
# gcg.choose()

# scg = SemanticConflictGenerator(model=model,prompt_template=PROMPT_TEMPLATE,max_retry=max_retry)
# scg.create()
# scg.choose()

