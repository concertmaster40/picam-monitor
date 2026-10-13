from flask import Flask, render_template, Response
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import threading

app = Flask(__name__)

# ==========================================
# CONFIGURAÇÕES DE E-MAIL (Preencha os dados)
# ==========================================
EMAIL_REMETENTE = "nickjackel@gmail.com"
SENHA_APP = "maxx rzda cytj xvat "
EMAIL_DESTINO = "nickjackel@gmail.com"

# ==========================================
# INICIALIZAÇÃO DA CÂMERA (Otimizado para Pi)
# ==========================================
# Força o uso do backend V4L2 (necessário no Linux/Raspberry Pi)
camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Ajusta a resolução para não sobrecarregar a CPU do Raspberry Pi 3
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Limita a taxa de captura nativa para poupar recursos
camera.set(cv2.CAP_PROP_FPS, 2)

# Aguarda 1 segundo para o sensor da câmera aquecer e estabilizar a luminosidade
time.sleep(1)

# Variável global para evitar envio de e-mails em massa
ultimo_alerta = 0 

def enviar_email(frame):
    """Gera o anexo e envia um email com a foto da detecção"""
    try:
        msg = MIMEMultipart()
        msg['Subject'] = 'ALERTA DE SEGURANÇA: Movimento na Sala de Servidores'
        msg['From'] = EMAIL_REMETENTE
        msg['To'] = EMAIL_DESTINO

        texto = MIMEText("Movimento detectado na sala de servidores. Verifique a imagem em anexo.")
        msg.attach(texto)

        # Converte o frame (imagem) para jpg e anexa
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            img_anexo = MIMEImage(jpeg.tobytes(), name="alerta.jpg")
            msg.attach(img_anexo)

        # Conexão SMTP (Exemplo usando Gmail. Para o da empresa, adapte host/porta)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_REMETENTE, SENHA_APP)
        server.send_message(msg)
        server.quit()
        print("E-mail de alerta enviado com sucesso!")
    except Exception as e:
        print(f"Erro ao enviar email: {e}")

def gerar_frames():
    global ultimo_alerta
    frame_anterior = None
    
    while True:
        sucesso, frame = camera.read()
        
        # Correção: Em vez de usar break, se falhar, aguarda um pouco e tenta novamente
        if not sucesso:
            time.sleep(0.1)
            continue

        # --- DETECÇÃO DE MOVIMENTO ---
        # Converte para escala de cinza e aplica desfoque (blur) para reduzir ruído visual
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Guarda o primeiro frame para base de comparação
        if frame_anterior is None:
            frame_anterior = gray
            continue

        # Calcula a diferença absoluta entre o frame atual e o anterior
        diferenca = cv2.absdiff(frame_anterior, gray)
        _, thresh = cv2.threshold(diferenca, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Encontra as áreas que sofreram alteração
        contornos, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        movimento_detectado = False
        for contorno in contornos:
            if cv2.contourArea(contorno) < 1000: # Ignora movimentos muito pequenos (ex: poeira/sombra leve)
                continue
            movimento_detectado = True
            
            # Desenha um retângulo vermelho onde houve movimento na tela
            (x, y, w, h) = cv2.boundingRect(contorno)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Atualiza o frame de referência
        frame_anterior = gray

        # Dispara e-mail se detectar movimento (Cooldown de 60 segundos entre disparos)
        tempo_atual = time.time()
        if movimento_detectado and (tempo_atual - ultimo_alerta > 60):
            ultimo_alerta = tempo_atual
            print("Movimento detectado! Iniciando envio de e-mail...")
            # Usa Threading para o envio de e-mail não congelar a imagem da web
            thread_email = threading.Thread(target=enviar_email, args=(frame.copy(),))
            thread_email.start()

        # --- STREAMING PARA O NAVEGADOR ---
        # Converte o quadro atual para JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Retorna o frame no formato Multipart exigido pelas tags <img> HTML para streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        # Controla o loop para exibir cerca de 2 FPS
        time.sleep(0.5) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Inicializa o servidor web (Acessível a partir de outras máquinas da rede Pumatronix)
    app.run(host='0.0.0.0', port=5000, debug=False)