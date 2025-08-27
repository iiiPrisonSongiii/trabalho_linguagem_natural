
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix

class ETL:
    def __init__(self, caminho_csv:str, qtd_max_de_tokens:int):
        self.caminho_csv = caminho_csv
        self.qtd_max_de_tokens = qtd_max_de_tokens
    
    def extracao_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.caminho_csv)
        return df
    
    # Basicamente aqui eu vou contar a quantidade de tokens
    # E também vou aplicar o filtro de quantidade máxima de tokens
    def tratamento(self, df:pd.DataFrame) -> pd.DataFrame:
        def limitar_tamanho_texto(text):
            toks = str(text).split()
            toks = toks[:self.qtd_max_de_tokens]
            toks = " ".join(toks)
            return toks

        df['preprocessed_news'] = df['preprocessed_news'].astype(str).apply(limitar_tamanho_texto)
        df['qtd_tokens'] = df['preprocessed_news'].str.split().apply(len)
        return df
    
class TreinamentoIARegressaoLogistica:
    def __init__(self, meu_ru:str, df:pd.DataFrame):
        self.meu_ru = meu_ru
        self.df = df
    
    # Separação dos arrays de treino e teste
    def segregacao_treino_teste(self, porcentagem_teste:float, organizacao_fixa:bool = None) -> dict: 
        texto = self.df['preprocessed_news'].values
        classificacao = self.df['label'].values
        
        if organizacao_fixa == True: organizacao_fixa = 20
        else: organizacao_fixa = None
            
        treino_texto, teste_texto, treino_classificacao, teste_classificacao = train_test_split(
            texto, classificacao, test_size=porcentagem_teste, random_state=organizacao_fixa,
            stratify=classificacao
        )
        return {
            'treino_texto': treino_texto,
            'teste_texto': teste_texto,
            'treino_classificacao': treino_classificacao,
            'teste_classificacao': teste_classificacao
        }
    
    # TF = Term Frequency → quantas vezes a palavra aparece no documento. 
    # IDF = Inverse Document Frequency → dá menos peso a palavras que 
    # aparecem em quase todos os textos (tipo “de”, “que”, “em”). 
    # Gera matrizes em tipo de objeto: csr_matrix
    def gerar_tf_idf(self, treino_texto:np.ndarray, teste_texto:np.ndarray,
                     qtd_n_grama:int, minimo_repeticao:int) -> dict:
        inst_vectorizer = TfidfVectorizer(ngram_range=(1,qtd_n_grama), min_df=minimo_repeticao)
        matriz_tf_idf_treino = inst_vectorizer.fit_transform(treino_texto)
        matriz_tf_idf_teste = inst_vectorizer.transform(teste_texto)
        termos = inst_vectorizer.get_feature_names_out()
        return {
            'matriz_tf_idf_treino': matriz_tf_idf_treino,
            'matriz_tf_idf_teste': matriz_tf_idf_teste,
            'termos': termos
        }
    
    # Aqui vamos testar a IA com os 25% destinados
    # Faço a predição e verifico a acurácia
    # Acurácia olha para dois arrays: Valores verdadeiros X Valores da predição
    def treinar(self, matriz_tf_idf_treino:csr_matrix,
                matriz_tf_idf_teste:csr_matrix, treino_classificacao:np.ndarray,
                teste_classificacao:np.ndarray, valor_maximo_tentativas:int) -> dict:
        inst_regressao = LogisticRegression(max_iter=valor_maximo_tentativas)
        inst_regressao.fit(matriz_tf_idf_treino, treino_classificacao)
        predicao = inst_regressao.predict(matriz_tf_idf_teste)

        acuracia = accuracy_score(teste_classificacao, predicao)
        return {'acuracia': acuracia}

# Criei uma classe responsável só para demonstração dos resultados
# Por gráfico de núvem de palavras
class Demonstracao:
    def __init__(self, meu_ru):
        self.cor_de_fundo = 'white'
        self.titulo = "AP - Ricardo Gomes - 4444964"
        self.meu_ru = meu_ru
        
    # tons de vermelho 
    def red_tones(*args, **kwargs):
        # saturação alta e luminosidade variando para criar tons diferentes
        return f"hsl(0, 90%, {np.random.randint(30,70)}%)"

    # tons de verde
    def green_tones(*args, **kwargs):
        return f"hsl(120, 90%, {np.random.randint(30,70)}%)"
    
    def gerar_grafico(self, filtro_classificacao:str,
                      treino_classificacao:np.ndarray, matriz_tf_idf_treino:csr_matrix, termos:np.ndarray):
        indice = np.where(treino_classificacao == filtro_classificacao)[0]
        media = np.asarray(matriz_tf_idf_treino[indice].mean(axis=0)).ravel()
        pesos = dict(zip(termos, media))
        
        # Inserindo de forma manual o meu RU dentro da imagem
        # Colocando como peso máximo, assim como diz o PDF de instuções
        pesos[self.meu_ru] = max(pesos.values()) * 2
        
        wc = WordCloud(background_color=self.cor_de_fundo).generate_from_frequencies(pesos)
        if filtro_classificacao == 'true':
            wc = wc.recolor(color_func=self.green_tones)
        else:
            wc = wc.recolor(color_func=self.red_tones)
        wc.to_file(f'{filtro_classificacao}.png')
        plt.imshow(wc)
        plt.axis("off")
        plt.title(self.titulo)
        plt.show()
    
if __name__ == '__main__':
    '''
    Oi professor.
    Neste espaço consta as variáveis de configuração,
    você pode mudar as configurações em um só lugar e testar,
    A princípio estou usando essas configurações abaixo
    '''
    # Definição do número máximo de palavras por texto
    # Faço isso para calibração, para que a IA não atribua
    # pesos excessívos para parâmetros de tamanho de texto
    qtd_max_de_tokens = 210
    # Caminho local do arquivo pre-processed.csv
    caminho_csv = "corpus/preprocessed/pre-processed.csv"
    meu_ru = '4444964'
    # Porcentagem destinada a teste, automaticamente 75% fica para treinamento
    porcentagem_teste = 0.25
    # Eu criei esse booleano para que a IA não distribua aleatoriamente
    # Os textos para treino e para texto, mas que sempre resulte na mesma distribuição
    organizacao_fixa = True
    # número máximo de tentativas de ajuste de pesos para encontrar o peso ideal
    valor_maximo_tentativas = 250
    # unigrama usar 1, bigramas usar 2, trigramas usar 3, e etc...
    # exemplo unigrama: “cachorro”, “caramelo”, “posto”, “policial”
    # exemplo bigramas: “cachorro caramelo”, “posto policial”, “garrafa água” 
    # exemplo trigramas: “Operação lava jato”, “Rio Grande Sul” 
    qtd_n_grama = 2
    # Mínimo de repetições de uma mesma palavra em todos os textos
    # Para que seja considerado no calculo dos pesos
    minimo_repeticao = 15
    
    # Classe destinada a extração, tratamento e carregamento dos dados
    # Retorna um dataframe
    inst_etl = ETL(
        qtd_max_de_tokens=qtd_max_de_tokens,
        caminho_csv=caminho_csv
    )
    df = inst_etl.extracao_csv()
    df = inst_etl.tratamento(df=df)
    
    
    inst_treino = TreinamentoIARegressaoLogistica(
        meu_ru=meu_ru,
        df=df
    )
    # Separação dos arrays de treino e teste
    dict_resultado = inst_treino.segregacao_treino_teste(
        porcentagem_teste=porcentagem_teste,
        organizacao_fixa=organizacao_fixa
    )
    # TF = Term Frequency → quantas vezes a palavra aparece no documento. 
    # IDF = Inverse Document Frequency → dá menos peso a palavras que 
    # aparecem em quase todos os textos (tipo “de”, “que”, “em”). 
    # Gera matrizes em tipo de objeto: csr_matrix
    dict_resultado|= inst_treino.gerar_tf_idf(
        treino_texto=dict_resultado['treino_texto'],
        teste_texto=dict_resultado['teste_texto'],
        qtd_n_grama=qtd_n_grama,
        minimo_repeticao=5
    )
    # Aqui vamos testar a IA com os 25% destinados
    # Faço a predição e verifico a acurácia
    # Acurácia olha para dois arrays: Valores verdadeiros X Valores da predição
    dict_resultado|= inst_treino.treinar(
        matriz_tf_idf_treino= dict_resultado['matriz_tf_idf_treino'],
        matriz_tf_idf_teste= dict_resultado['matriz_tf_idf_teste'],
        treino_classificacao= dict_resultado['treino_classificacao'],
        teste_classificacao= dict_resultado['teste_classificacao'],
        valor_maximo_tentativas= valor_maximo_tentativas
    )
    
    
    # Criei uma classe responsável só para demonstração dos resultados
    # Por gráfico de núvem de palavras
    inst_demonstracao = Demonstracao(meu_ru=meu_ru)
    
    # Gerando gráfico das notícias Verdadeiras e salvando:
    inst_demonstracao.gerar_grafico(
        filtro_classificacao='true',
        treino_classificacao= dict_resultado['treino_classificacao'],
        matriz_tf_idf_treino=dict_resultado['matriz_tf_idf_treino'],
        termos=dict_resultado['termos']
    )
    # Gerando gráfico das notícias Falsas e salvando:
    inst_demonstracao.gerar_grafico(
        filtro_classificacao='fake',
        treino_classificacao= dict_resultado['treino_classificacao'],
        matriz_tf_idf_treino=dict_resultado['matriz_tf_idf_treino'],
        termos=dict_resultado['termos']
    )