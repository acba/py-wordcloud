import pandas as pd
import pdfplumber
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import base64

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('omw')
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('portuguese')

def get_text(path):
    texto_completo = ''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pg_texto = page.extract_text()
            pg_num = page.page_number
            texto_completo+= ' ' + pg_texto
    return texto_completo

def preprocess_text(texto):
    # Tokenise words while ignoring punctuation
    tokeniser = RegexpTokenizer(r'[a-zA-Z0-9]+')
    tokens = tokeniser.tokenize(texto)

    tokens = [token.lower() for token in tokens]

    # remove stopwords
    keywords = [token for token in tokens if token not in nltk_stopwords]
    return keywords


texto = get_text('dados/edital_de_abertura_n_001_2020.pdf')

# lista de stopword
stopwords = set(STOPWORDS)
stopwords.update(['a', 'as', 'e', 'o', 'os', 'da', 'do', 'das', 'dos', 'de', 'eu', 'tu', 'em', 'ao', 'para', "meu", "você" ])
stopwords.update(nltk_stopwords)

wc = WordCloud(stopwords=stopwords, background_color="white", width=1600, height=800, collocations=False)
wc.generate(texto)

plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
# plt.show()

file_png = BytesIO()
plt.savefig(file_png, format='png')
file_png.seek(0)  # rewind to beginning of file
figdata_png = base64.b64encode(file_png.getvalue()).decode('utf-8')

plt.savefig(f'saida/wordcloud.svg', format='svg', dpi=1200)
plt.savefig(f'saida/wordcloud.png', format='png', dpi=1200)


keywords = preprocess_text(texto)

# # Create an instance of TfidfVectorizer
# vectoriser = TfidfVectorizer(analyzer=preprocess_text)
# # Fit to the data and transform to feature matrix
# X_train = vectoriser.fit_transform(texto)
# # Convert sparse matrix to dataframe
# X_train = pd.DataFrame.sparse.from_spmatrix(X_train)
# # Save mapping on which index refers to which words
# col_map = {v:k for k, v in vectoriser.vocabulary_.items()}
# # Rename each column using the mapping
# for col in X_train.columns:
#     X_train.rename(columns={col: col_map[col]}, inplace=True)
# X_train
