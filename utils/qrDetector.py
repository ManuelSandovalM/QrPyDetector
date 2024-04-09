import numpy as np
import tensorflow.lite as tflite

IMAGE_SIZE = 320

class ModelInference:

    def __init__(self, modelPath, resizeTo = IMAGE_SIZE) -> None:
        # Carga el modelo TFLite
        self.interpreter = tflite.Interpreter(model_path=modelPath)
        self.interpreter.allocate_tensors()

        # Obtiene los detalles del modelo
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        #print(self.input_details)
        self.image = None
        self.resizeTo = resizeTo

    def runDetection(self, image, labels=None):
        #try:
        self.image = image
        input_data = preProcessTF(self.image, self.resizeTo)

        # Asigna los datos de entrada al intérprete
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Ejecuta la inferencia
        self.interpreter.invoke()

        output_index1 = 600 #scores
        output_index2 = 598 #bounding_boxes
        output_index3 = 601 #???
        output_index4 = 599 #Classes

        scores = self.interpreter.get_tensor(output_index1)[0]
        boxes = self.interpreter.get_tensor(output_index2)[0]
        #output_data3 = self.interpreter.get_tensor(output_index3)[0]
        classes = self.interpreter.get_tensor(output_index4)[0]
        
        
        confidence_threshold = 0.1
        
        valid_scores = [scores[i] for i in range(len(scores)) if scores[i] > confidence_threshold]
        valid_boxes = [boxes[i] for i in range(len(scores)) if scores[i] > confidence_threshold]
        valid_classes = [classes[i] for i in range(len(scores)) if scores[i] > confidence_threshold]


        #Escala los bounding Boxes
        for i in range(len(valid_boxes)):
            ymin, xmin, ymax, xmax = valid_boxes[i]
            width, height = self.image.size
            x_min = int(xmin * width)
            x_max = int(xmax * width)
            y_min = int(ymin * height)
            y_max = int(ymax * height)

            if x_min<0:
                x_min = 1
            
            if y_min<0:
                y_min = 1
            
            if x_max>width:
                x_max = width-2
            
            if y_max>height:
                y_max = height-2

            valid_boxes[i] = (x_min, y_min, x_max, y_max)


        #Normaliza los arreglos
        valid_scores = np.array(valid_scores)
        valid_boxes = np.array(valid_boxes)
        valid_classes = np.array(valid_classes)

        #Filtra los objetos con más de una detección duplicada
        score_threshold = 0.4
        iou_threshold = 0.4

        selected_indices = non_max_suppression_msm(valid_boxes, valid_scores, score_threshold, iou_threshold )

        valid_scores = valid_scores[selected_indices]
        valid_boxes = valid_boxes[selected_indices]
        valid_classes = valid_classes[selected_indices]

        #Genera la lista de Detecciones final
        detections = []
        for i in range(len(valid_scores)):
            box = valid_boxes[i]
            rect_boxes = Rect(box[0], box[1], box[2], box[3])

            score = valid_scores[i]
            boundingBox = rect_boxes
            labelIndex = int(valid_classes[i])

            if labels:
                label = labels[labelIndex]
            else:
                label = None

            detections.append(Detection(boundingBox, score, labelIndex, label))
        
        return 0, detections
    

class QRDetector:
    def __init__(self, modelPath = './utils/lib/qrsDetection.tflite') -> None:
        MODEL_PATH = modelPath

        self.modelInference = ModelInference(MODEL_PATH, 320)

        self.labels = ["Blurred_QR", "QR"]

    def detect(self, image):
        status, response = self.modelInference.runDetection(image, self.labels)
        return status, response
    
    def drawBoundingBox(self, image, boundingBox):
        # Coordenadas del bounding box (izquierda, superior, derecha, inferior)
        rectangle = (boundingBox.left, boundingBox.top, boundingBox.right, boundingBox.bottom)

        return self.drawRectangle(image, rectangle)


class Rect:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top
    
    def getCenterX(self):
        return self.left + (self.width()/2)
    
    def getCenterY(self):
        return self.top + (self.height()/2)
    
    def getArea(self):
        return self.width()*self.height()
    
    def asArray(self):
        return [self.left, self.top, self.right, self.bottom]
    

class Detection:
    def __init__(self, boundingBox, score, labelIndex, label=None):
        self.boundingBox = boundingBox
        self.score = score
        self.labelIndex = labelIndex
        self.label = label

    def getBoundingBox(self):
        return self.boundingBox
    
    def getScore(self):
        return self.score
    
    def getLabelIndex(self):
        return self.labelIndex
    
    def getLabel(self):
        return self.label
    

def non_max_suppression_msm(bounding_boxes, scores, score_threshold, iou_threshold):
    
    picked_indices = []
    filter_scores = [index for index, score in enumerate(scores) if score > score_threshold]

    sorted_indices = sorted(filter_scores, key = lambda index: scores[index])

    while sorted_indices:
        last_index = len(sorted_indices) - 1
        picked_index = sorted_indices[last_index]
        picked_indices.append(picked_index)

        i = 0

        while i < last_index:
            current_index = sorted_indices[i]
            current_box = bounding_boxes[current_index]
            picked_box = bounding_boxes[picked_index]

            overlap = calculate_overlap(current_box, picked_box)

            if overlap > iou_threshold:
                sorted_indices.pop(i)
                last_index -= 1
            else:
                i += 1

        sorted_indices.pop()

    return picked_indices
    

def calculate_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    area_iou = max(0, x2 - x1) * max(0, y2 - y1)

    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    overlap = area_iou // (b1_area + b2_area - area_iou)

    return overlap


def preProcessTF(image, resize_size):

    image = image.resize((resize_size, resize_size))
    np_image = np.array(image, dtype=np.uint8)
    
    return np.expand_dims(np_image, axis=0)