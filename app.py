import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

df = pd.read_csv('database/processed/movies_cleaned.csv')
df = df.dropna(subset=['text_for_ia']).reset_index(drop=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatOllama(model="llama3.2", temperature=0.7)

matriz_vetores = model.encode(df['text_for_ia'].tolist(), show_progress_bar=True)

print("Calculando similaridade semântica")
simularidade = cosine_similarity(matriz_vetores)

def recomendar_filme(nome_filme, top_k=5):

  filme = df[df['title'].str.lower() == nome_filme.lower()] 

  if filme.empty: 
    return f"Filme {nome_filme} não existe no banco de dados!"
  
  idx = filme.index[0]
  scores = list(enumerate(simularidade[idx]))

  scores_ordenados = sorted(scores, key=lambda x: x[1], reverse=True)
    
  top_indices = [i[0] for i in scores_ordenados[1:top_k+1]]
    
  return df['title'].iloc[top_indices].tolist()


def get_similares (nome_filme):

  return recomendar_filme(nome_filme)

def genai_resposta(filme_user):

  recomendacoes = get_similares(filme_user)

  if isinstance(recomendacoes, str):
    return recomendacoes
  
  template = """
    Você é um assistente especialista em cinema.
    O usuário disse que gosta do filme: {filme_input}.
    
    Baseado na nossa análise de dados, encontramos estes filmes similares para ele:
    {lista_filmes}
    
    Crie uma resposta curta e engajadora recomendando esses filmes. 
    Não invente filmes que não estão na lista.
    """
    
  prompt = ChatPromptTemplate.from_template(template)
    
  chain = prompt | llm
    
  resposta = chain.invoke({
        "filme_input": filme_user,
        "lista_filmes": ", ".join(recomendacoes) 
    })
    
  return resposta.content

print("Consultando Llama...")
print(genai_resposta("The Dark knight"))