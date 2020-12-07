import pandas as pd
import matplotlib.pyplot as plt

import utils

utils.init_nltk()

texto = utils.get_text('dados/DiarioOficialMPPB-2020-12-01.pdf')
texto = texto.replace('\n', ' ')
texto = utils.remove_stopwords(texto)

informacoes = utils.gera_info(texto)

wc = utils.generate_wordcloud(texto)

wc_b64 = utils.wc2base64(wc)
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")

plt.savefig(f'saida/wordcloud.svg', format='svg', dpi=1200)


keywords = utils.preprocess_text(texto)
