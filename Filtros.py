import sys

from ParticleFilter import ParticleFilter
from KalmanFilter import KalmanFilter

import numpy as np
import cv2
import time
import mediapipe as mp
import random
from google.protobuf.json_format import MessageToDict
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

def find_hands(frame, hand_detector, show=True):

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    frame.flags.writeable = False
    results = hand_detector.process(frame)
    frame.flags.writeable = True

    hands_coord = []
    hands_handedness = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Mostrar esqueleto de las manos
            if show:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Guardar coordenadas de las manos
            hands_coord.append(np.array([
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame.shape[1],
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame.shape[0]], dtype='float32'))
                
    # Guardar si cada mano es izquierda o derecha
    if results.multi_handedness:
        for handedness in results.multi_handedness:
            handedness_dict = MessageToDict(handedness)
            whichHand = (handedness_dict['classification'][0]['label'])
            hands_handedness.append(whichHand)

    return [hands_coord, hands_handedness]

# Usar el filtro de partículas
def apply_particles(in_path, out_path, numParticles, mean, variance, show_boxes = True):

    # Máscara para la galleta
    template = cv2.imread('./images/cookie80.png')
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    template_mask = np.zeros(shape=template.shape)
    template_mask[:,:,0] = template_gray < 250
    template_mask[:,:,1] = template_gray < 250
    template_mask[:,:,2] = template_gray < 250
    cap = cv2.VideoCapture(in_path) # fichero a procesar

    if (cap.isOpened()):

        # leemos el primer frame
        ret, frame = cap.read()
        
        # Flip frame 
        frame = cv2.flip(frame, 1)

        # creamos el video de salida
        if out_path != None:
            height, width, layers = frame.shape
            size = (width, height)
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        # Inicialización de la clase filtro de partículas
        ParticleFilter.create(numParticles)

        # Inicializar de forma aleatoria
        for p in ParticleFilter.particles:
            p.x = random.randrange(100, frame.shape[1] - 100)
            p.y = random.randrange(100, frame.shape[0] - 100)
        
        # Inicialización del tiempo
        pTime = 0

        # Inicialización del detector de manos
        hand_detector = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # mientras se haya podido leer el siguiente frame
        while(cap.isOpened() and ret):

            # Salir
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q') or k == 27:
                break

            # Buscar las manos
            hands = find_hands(frame, hand_detector, show=show_boxes)

            # medimos las particulas
            ParticleFilter.measure(hands)

            # dibujamos todas las particulas
            if show_boxes:
                for p in ParticleFilter.particles:
                    p.draw(frame, (0,0, 100 + 155*p.weight))

            # Poner galleta
            x = np.clip(ParticleFilter.bestParticle.x, int(template.shape[1]/2), frame.shape[1] - int(template.shape[1]/2))
            y = np.clip(ParticleFilter.bestParticle.y, int(template.shape[0]/2), frame.shape[0] - int(template.shape[0]/2))
            np.putmask(frame[int(y-template.shape[0]/2):int(y+template.shape[0]/2), int(x-template.shape[1]/2):int(x+template.shape[1]/2), :], template_mask, template)

            # Texto Ganador
            cv2.putText(frame, ParticleFilter.bestParticle.ganador, (int(frame.shape[1]/2) - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, f'FPS:{int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Mostrar frame
            cv2.imshow('frame', frame)

            # Guardar frame
            if out_path is not None:
                out.write(frame)

            # preparamos para el siguiente frame, remuestrear y simular
            ParticleFilter.resample()
            ParticleFilter.simulate(mean, variance)

            # leer siguiente frame
            ret, frame = cap.read()

            # Flip frame 
            frame = cv2.flip(frame, 1)


        if out_path != None:
            out.release()

    cap.release()
    cv2.destroyAllWindows()

def apply_kalman(in_path, out_path, show_boxes = True, dst_max = 50):

    # Máscara para la galleta
    template = cv2.imread('./images/cookie80.png')
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    template_mask = np.zeros(shape=template.shape)
    template_mask[:,:,0] = template_gray < 250
    template_mask[:,:,1] = template_gray < 250
    template_mask[:,:,2] = template_gray < 250
    cap = cv2.VideoCapture(in_path) # fichero a procesar

    if (cap.isOpened()):

        # leemos el primer frame
        ret, frame = cap.read()
        
        # Flip frame 
        frame = cv2.flip(frame, 1)

        # creamos el video de salida
        if out_path != None:
            height, width, layers = frame.shape
            size = (width, height)
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        # Inicialización del tiempo
        pTime = 0

        # Inicialización del detector de manos
        hand_detector = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # mientras se haya podido leer el siguiente frame
        while(cap.isOpened() and ret):

            # Salir
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q') or k == 27:
                break

            # Buscar las manos
            hands = find_hands(frame, hand_detector, show=show_boxes)
            
            # Predict
            (x, y) = KF.predict()
            x = int(x)
            y = int(y)
            
            # Determinar quina mà està més aprop
            txt = "Empat!"
            coord_guanyador = np.array([[x],[y]])

            if len(hands[0]) > 0 and len(hands[0][0]) > 0:

                position = np.array([x, y])
                h_coord, h_handedness = hands

                if len(h_coord) == 2 and len(h_handedness) == 2:
                    dst_1 = np.linalg.norm(position - h_coord[0])
                    dst_2 = np.linalg.norm(position - h_coord[1])

                    if dst_1 < dst_2:
                        coord_guanyador = np.array([[h_coord[0][0]],[h_coord[0][1]]])   
                        txt = "Guanya: " + h_handedness[0]

                    elif dst_1 > dst_2:
                        coord_guanyador = np.array([[h_coord[1][0]],[h_coord[1][1]]]) 
                        txt = "Guanya: " + h_handedness[1]
                
                elif len(h_coord) == 1 and len(h_handedness) == 1:
                    coord_guanyador = np.array([[h_coord[0][0]],[h_coord[0][1]]]) 
                    txt = "Guanya: " + h_handedness[0]
                
            # Update: si no es troben mans, s'actualitza amb predicció anterior
            (x1, y1) = KF.update(coord_guanyador)
            x1 = int(x1)
            y1 = int(y1)

            # Poner galleta
            x = np.clip(int(x1), int(template.shape[1]/2), frame.shape[1] - int(template.shape[1]/2))
            y = np.clip(int(y1), int(template.shape[0]/2), frame.shape[0] - int(template.shape[0]/2))
            np.putmask(frame[int(y-template.shape[0]/2):int(y+template.shape[0]/2), int(x-template.shape[1]/2):int(x+template.shape[1]/2), :], template_mask, template)


            # # Texto Ganador
            cv2.putText(frame, txt, (int(frame.shape[1]/2) - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, f'FPS:{int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Mostrar frame
            cv2.imshow('frame', frame)

            # Guardar frame
            if out_path is not None:
                out.write(frame)

            # leer siguiente frame
            ret, frame = cap.read()

            # Flip frame 
            frame = cv2.flip(frame, 1)

        if out_path != None:
            out.release()

    cap.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Apply Kalman or Particle filter.')
parser.add_argument('--filter', dest='filter', type=str, default='kalman',
                    help='Chosen filter.')
parser.add_argument('--out_path', dest='out_path', type=str, default=None,
                    help='Out path of video.')
parser.add_argument('--show_boxes', dest='show_boxes', action='store_true', default=False,
                    help='Either show boxes of the detections along with hand skeletons or not.')

args = parser.parse_args()

if args.filter == 'kalman':
    apply_kalman(in_path=0, out_path=args.out_path, show_boxes=args.show_boxes)
else:
    apply_particles(in_path=0, out_path=args.out_path, numParticles = 300, mean=0, variance=10, show_boxes=args.show_boxes)