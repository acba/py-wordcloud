import pdfplumber
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import re
import base64

import nltk
from nltk.tokenize import RegexpTokenizer

NLTK_PT = nltk.corpus.stopwords.words('portuguese')
STOPWORDS_EXTRA = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'x', 'w', 'y', 'z'
]

def init_nltk():
    nltk.download('rslp')
    nltk.download('stopwords')
    nltk.download('omw')

def extract_pdf(path):
    texto_completo = ''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pg_texto = page.extract_text()
            pg_num = page.page_number
            texto_completo+= ' ' + pg_texto

    return texto_completo

def get_filetype(path):
    return 'application/pdf'

def get_text(path):    
    if get_filetype(path) == 'application/pdf':
        return extract_pdf(path)
    else:
        print('Formato nao suportado')


def remove_stopwords(texto):
    tratado = [token.lower() for token in texto.split() if token not in NLTK_PT and token not in STOPWORDS_EXTRA]
    return ' '.join(tratado)

def preprocess_text(texto):
    # Tokenise words while ignoring punctuation
    # tokeniser = RegexpTokenizer(r'[a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ0-9]+')
    tokeniser = RegexpTokenizer(r'[a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ./-@]+')
    tokens = tokeniser.tokenize(texto)


    # tokens = [token.lower() for token in texto.split() if token not in NLTK_PT]


    tokens = [token.lower() for token in tokens]

    # remove stopwords
    keywords = [token for token in tokens if token not in NLTK_PT and token not in STOPWORDS_EXTRA]
    return keywords

def generate_wordcloud(texto):

    # lista de stopwords
    stopwords = set(STOPWORDS)
    stopwords.update(NLTK_PT)
    stopwords.update(STOPWORDS_EXTRA)

    wc = WordCloud(stopwords=stopwords, background_color="white", width=1600, height=800, collocations=False)
    wc.generate(texto)

    return wc

def wc2base64(wc):
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")

    arquivo = BytesIO()
    plt.savefig(arquivo, format='png', dpi=1200)
    arquivo.seek(0)
    return base64.b64encode(arquivo.getvalue()).decode('utf-8')


def filter_cpf(texto):
    regra  = '([0-9]{3}[\.]?[0-9]{3}[\.]?[0-9]{3}[-]?[0-9]{2})'
    encontrados = re.findall(regra, texto)

    return encontrados


def filter_cnpj(texto):
    regra = '([0-9]{2}\.?[0-9]{3}\.?[0-9]{3}\/?[0-9]{4}\-?[0-9]{2})'
    encontrados = re.findall(regra, texto)

    return encontrados
    

def filter_email(texto):
    regra = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    encontrados = re.findall(regra, texto)

    return encontrados


def filter_telefone(texto):
    regra = '(\(?\d{2}\)?\s?\d{4,5}\-\d{4})'
    encontrados = re.findall(regra, texto)

    return encontrados

def filter_urls(texto):
    regra = '(https?:\/\/)?(www\.)[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)|(https?:\/\/)?(www\.)?(?!ww)[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    encontrados = re.findall(regra, texto)

    return encontrados


def gera_info(texto):
    cpfs = filter_cpf(texto)
    cnpjs = filter_cnpj(texto)
    emails = filter_email(texto)
    telefones = filter_telefone(texto)
    urls = filter_urls(texto)

    tokens = remove_stopwords(texto).split(' ')
    mais_frequentes = nltk.FreqDist(tokens).most_common(20)

    return { 'cpfs': cpfs, 'cnpjs': cnpjs, 'emails': emails, 'telefones': telefones, 'urls': urls, 'mais_frequentes': mais_frequentes}