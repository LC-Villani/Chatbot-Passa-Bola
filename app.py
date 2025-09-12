# app.py

from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Inicializa o aplicativo Flask
app = Flask(__name__)


# --- Base de Conhecimento (Perguntas e Respostas) para o App Passa-Bola ---
knowledge_base = {
    # --- 1. Sobre o App ---
    "O que é o Passa-Bola?": "O Passa-Bola é uma plataforma dedicada a conectar mulheres que amam futebol! Aqui você pode encontrar e marcar lugares para jogar, criar eventos, compartilhar seus melhores lances e até ser descoberta por clubes.",
    "Para quem é este aplicativo?": "O app é para todas as mulheres que jogam ou se interessam por futebol, desde a iniciante que busca um time casual até a jogadora experiente que quer mais visibilidade.",
    "O app é gratuito?": "Sim, o uso principal do Passa-Bola, como encontrar locais e participar de eventos, é totalmente gratuito.",

    # --- 2. Funcionalidades ---
    "Como faço para marcar um lugar para jogar?": "Na seção 'Jogar', você pode usar o mapa para encontrar campos e quadras próximas. Ao encontrar um local, você pode ver os horários disponíveis e criar um novo evento ou entrar em um já existente.",
    "O que são os eventos?": "Eventos são as partidas! Você pode criar um evento público, aberto para qualquer jogadora do app, ou um evento privado, apenas para as amigas que você convidar.",
    "Como funciona para postar meus lances?": "Pense nisso como uma rede social para jogadoras! Na sua timeline, você pode postar vídeos ou fotos dos seus treinos, gols e jogadas, como se fosse um Twitter ou Instagram do futebol.",
    "Como os clubes e instituições usam o app?": "Clubes, olheiros e instituições podem criar um perfil verificado. Eles usam a plataforma para divulgar peneiras, torneios e até mesmo para observar o talento de jogadoras ativas no app.",
    "Posso ser contratada por um time usando o app?": "Sim, essa é uma das nossas metas! Ao manter seu perfil atualizado com seus lances e informações, você aumenta sua visibilidade para os olheiros e clubes que usam o Passa-Bola para descobrir novos talentos.",

    # --- 3. Futebol Feminino (Engajamento) ---
    "Quando é o próximo jogo da seleção brasileira feminina?": "Para saber as datas exatas dos próximos jogos, recomendo checar o calendário oficial da CBF. A seleção está sempre em preparação para competições importantes!",
    "Quem é a maior artilheira da história da seleção?": "A rainha Marta! Ela é a maior artilheira de todas as seleções brasileiras (masculina e feminina) e uma inspiração para todas nós.",
    "Onde posso assistir aos jogos do campeonato brasileiro feminino?": "Os direitos de transmissão variam, mas muitos jogos são transmitidos por canais de TV aberta como a Band, canais por assinatura como o SporTV e também por serviços de streaming.",
    
    # --- 4. Suporte e Ajuda ---
    "Esqueci minha senha, o que eu faço?": "Na tela de login, clique em 'Esqueci minha senha' e siga as instruções para criar uma nova senha através do seu e-mail cadastrado.",
    "Como posso denunciar um perfil ou post inadequado?": "A segurança da nossa comunidade é prioridade. Em cada perfil e post, você encontrará uma opção (geralmente três pontinhos) para denunciar. Nossa equipe de moderação analisará o caso.",
    "Encontrei um bug no aplicativo, como reporto?": "Ficamos gratas pelo seu feedback! Por favor, envie um e-mail para suporte@passabola.com.br com os detalhes do problema e, se possível, uma captura de tela.",

    # --- 5. Interações Sociais ---
    "oi": "Olá! Sou a assistente do Passa-Bola. Como posso te ajudar a entrar em campo hoje?",
    "olá": "Olá! Sou a assistente do Passa-Bola. Como posso te ajudar a entrar em campo hoje?",
    "tudo bem?": "Tudo ótimo por aqui, pronta pra ajudar a bola rolar! E com você?",
    "obrigado": "De nada! Se precisar de mais alguma coisa, é só chamar.",
    "obrigada": "De nada! Se precisar de mais alguma coisa, é só chamar.",
    "tchau": "Até mais! Bons jogos!"
}

# Extrai as perguntas da base de conhecimento
perguntas = list(knowledge_base.keys())

# --- Lógica do Chatbot com NLP ---

# Vetorização das perguntas

# TfidfVectorizer transforma o texto em vetores numéricos que o computador pode entender.
vectorizer = TfidfVectorizer(ngram_range=(1,2)) # ngram_range considera palavras e pares de palavras
vetores_perguntas = vectorizer.fit_transform(perguntas) # Cria os vetores para todas as perguntas

# Função para encontrar a melhor resposta

def encontrar_melhor_resposta(mensagem_usuario):
    # Transforma a mensagem do usuário no mesmo tipo de vetor das perguntas
    vetor_usuario = vectorizer.transform([mensagem_usuario])
    
    # Calcula a similaridade de cosseno entre a pergunta do usuário e todas as perguntas da base
    similaridades = cosine_similarity(vetor_usuario, vetores_perguntas)
    
    # Encontra o índice da pergunta mais similar
    indice_mais_similar = np.argmax(similaridades)
    
    # Define uma divisa de confiança. Se a similaridade for muito baixa, não responde.
    limiar_confianca = 0.3 
    if similaridades[0][indice_mais_similar] > limiar_confianca:
        return knowledge_base[perguntas[indice_mais_similar]]
    else:
        return "Desculpe, não entendi sua pergunta. Poderia tentar reformulá-la?"

# --- Criação da API ---

@app.route('/chat', methods=['POST'])
def chat():
    # Teoricamente pega a mensagem do usuário que veio do frontend  (Teoricamente pois o frontend ainda não existe)
    data = request.json 
    mensagem_usuario = data.get('message') # Pega a mensagem enviada pelo usuário
    
    if not mensagem_usuario:
        return jsonify({"error": "Nenhuma mensagem recebida"}), 400 # Retorna erro se não houver mensagem
        
    # Obtém a resposta do chatbot
    resposta_bot = encontrar_melhor_resposta(mensagem_usuario) # Chama a função para encontrar a melhor resposta
    
    # Retorna a resposta em formato JSON
    return jsonify({"response": resposta_bot}) # Retorna a resposta do bot

# Roda o servidor
if __name__ == '__main__':
    app.run(debug=True)