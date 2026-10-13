from flask import Flask, render_template, Response
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
import threading
import credentials as cred

app = Flask(__name__)

# Configurações de E-mail
# EMAIL_REMETENTE = "seu_email@gmail.com"
# SENHA_APP = "sua_senha_de_app_gmail"
# EMAIL_DESTINO = "ti@pumatronix.com"

# Variáveis globais para controle
camera = cv2.VideoCapture(0) # 0 para a câmera padrão conectada (Pi Cam ou USB)
camera.set(cv2.CAP_PROP_FPS, 2) # Limita a câmera a 2 FPS
ultimo_alerta = 0 # Tempo do último alerta enviado

def enviar_email(frame):
    """Envia um email com a foto da detecção de movimento"""
    try:
        msg = MIMEMultipart()
        msg['Subject'] = 'ALERTA DE SEGURANÇA: Movimento na Sala de Servidores'
        msg['From'] = cred.EMAIL_REMETENTE
        msg['To'] = cred.EMAIL_DESTINO

        texto = MIMEText("Movimento detectado na sala de servidores. Verifique a imagem em anexo.")
        msg.attach(texto)

        # Converte o frame (imagem) para jpg e anexa
        ret, jpeg = cv2.imencode('.jpg', frame)
        img_anexo = MIMEImage(jpeg.tobytes(), name="alerta.jpg")
        msg.attach(img_anexo)

        # Conexão SMTP (Exemplo usando Gmail)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(cred.EMAIL_REMETENTE, cred.SENHA_APP)
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
        if not sucesso:
            break
        else:
            # --- DETECÇÃO DE MOVIMENTO ---
            # Converte para escala de cinza e aplica um blur para reduzir ruído
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if frame_anterior is None:
                frame_anterior = gray
                continue

            # Calcula a diferença absoluta entre o frame atual e o anterior
            diferenca = cv2.absdiff(frame_anterior, gray)
            _, thresh = cv2.threshold(diferenca, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contornos, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            movimento_detectado = False
            for contorno in contornos:
                if cv2.contourArea(contorno) < 1000: # Ignora movimentos muito pequenos
                    continue
                movimento_detectado = True
                # Desenha um retângulo onde houve movimento
                (x, y, w, h) = cv2.boundingRect(contorno)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Atualiza o frame anterior
            frame_anterior = gray

            # Envia email se detectar movimento (com cooldown de 60 segundos para não inundar a caixa de entrada)
            tempo_atual = time.time()
            if movimento_detectado and (tempo_atual - ultimo_alerta > 60):
                ultimo_alerta = tempo_atual
                print("Movimento detectado! Iniciando envio de e-mail...")
                # Inicia uma thread para enviar o e-mail sem travar o vídeo
                thread_email = threading.Thread(target=enviar_email, args=(frame.copy(),))
                thread_email.start()

            # --- STREAMING DA WEB ---
            # Converte de volta para JPEG para mostrar no HTML
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Formato multipart para streaming contínuo no navegador
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
            time.sleep(0.5) # Força rodar a aproximadamente 2 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Roda o servidor acessível na rede local na porta 5000
    app.run(host='0.0.0.0', port=5000, debug=False)