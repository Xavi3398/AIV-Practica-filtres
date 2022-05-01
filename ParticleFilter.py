import numpy as np
import cv2
import copy

# clase que implementa el Filtro de Particulas
class ParticleFilter(object):
    def __init__(self):
        # datos de una particula
        # INI_MODIFICABLE
        self.x = 0
        self.y = 0
        # FIN_MODIFICABLE

        # atributos necesarios para llevar a cabo el filtro de particulas
        self.weight = 0
        self.initWeight = 0
        self.acumWeight = 0
        self.ganador = 'Empate!'

    def simulateParticle(self, mean, variance):
        # Aplica ruido gaussiano
        self.x += np.random.normal(mean, variance)
        self.y += np.random.normal(mean, variance)

    def esError(self, hands, position):

        h_coord, h_handedness = hands

        txt = "Empat!"
        error = ParticleFilter.dst_max

        if len(h_coord) == 2 and len(h_handedness) == 2:
            dst_1 = np.linalg.norm(position - h_coord[0])
            dst_2 = np.linalg.norm(position - h_coord[1])

            if dst_1 < dst_2 and dst_1 < ParticleFilter.dst_max:
                txt = "Guanya: " + h_handedness[0]
                error = dst_1

            elif dst_1 > dst_2 and dst_2 < ParticleFilter.dst_max:
                txt = "Guanya: " + h_handedness[1]
                error = dst_2
        
        elif len(h_coord) == 1 and len(h_handedness) == 1:
            dst = np.linalg.norm(position - h_coord[0])
            if dst < ParticleFilter.dst_max:
                txt = "Guanya: " + h_handedness[0]
                error = dst
        
        return error, txt

    def measureParticle(self, hands):
        # medimos el error y le asignamos un peso
        error, txt = self.esError(hands, np.array([self.x, self.y], dtype='float32'))
        self.weight = (1 - error/ParticleFilter.dst_max)*0.99 + 0.01
        self.ganador = txt
        # self.weight = (1 - error/total)*0.99 + 0.01




    # NO MODIFICAR NADA
    # atributos para llevar a cabo el filtro de particulas
    particles = []
    bestParticle = None
    totalWeight = 0
    dist_encert = 10
    dst_max = 50
    
    min_pinsa = [20, 52, 32]
    max_pinsa = [50, 255, 236]

    min_etiq = [60, 100, 24]
    max_etiq = [80, 255, 221]

    # crea las particulas
    @staticmethod
    def create(num):
        for x in range(num):
            ParticleFilter.particles.append(ParticleFilter())

    # simula todas las particulas
    @staticmethod
    def simulate(mean, variance):
        for p in ParticleFilter.particles:
            p.simulateParticle(mean, variance)

    # realiza la medicion de cada particula, calculando el peso inicial y final de cada particula
    # la mejor particula
    @staticmethod
    def measure(hands):
        ParticleFilter.bestWeight = 0
        ParticleFilter.bestParticle = None
        ParticleFilter.totalWeight = 0
        for p in ParticleFilter.particles:
            p.initWeight = ParticleFilter.totalWeight
            p.measureParticle(hands)
            w = p.weight
            if w > ParticleFilter.bestWeight:
                ParticleFilter.bestWeight = w
                ParticleFilter.bestParticle = p
            ParticleFilter.totalWeight += w
            p.acumWeight = ParticleFilter.totalWeight

    # remuestrea las particulas teniendo en cuenta el peso de cada una
    @staticmethod
    def resample():
        # se generan los numeros aleatorios
        weights = np.random.uniform(0, ParticleFilter.totalWeight, len(ParticleFilter.particles))
        new_particles = []
        for w in weights:
            # se obtiene la particula y se introduce una copia
            p = ParticleFilter.getParticleByWeight(w)
            new_particles.append(copy.copy(p))

        ParticleFilter.particles = new_particles

    # devuelve la particula que tiene asignado el intervalo que contiene a w
    @staticmethod
    def getParticleByWeight(w):
        inf = 0
        sup = len(ParticleFilter.particles)
        while (inf <= sup):
            centro = (inf+sup)//2
            p = ParticleFilter.particles[centro]
            if (w < p.initWeight):
                sup = centro - 1
            elif w > p.acumWeight:
                inf = centro + 1
            else:
                return p

        return None

    # dibuja una particula como un cuadrado de color
    def draw(self, image, color):
        cv2.rectangle(image, (int(self.x-5), int(self.y-5)), (int(self.x+5), int(self.y+5)), color, 1)