import sys
import time
import colorsys


from PIL import Image, ImageQt
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body, yolo_eval_mine1, yolo_eval_mine2
from yolo3.utils import letterbox_image
import os

from PyQt5 import QtWidgets, Qt, QtGui, QtCore
from PyQt5.QtWidgets import *
import cv2
from timeit import default_timer as timer
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator, QFont
from PyQt5.QtCore import pyqtSignal

from tools import generate_detections as gdet
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from PyQt5.QtCore import Qt

import winsound
import platform

# setting widget
class settings(QtWidgets.QWidget):

    si1 = pyqtSignal(str)
    io1 = pyqtSignal(str)
    sc1 = pyqtSignal(str)
    nm1 = pyqtSignal(int)
    tr1 = pyqtSignal(int)
    mo1 = pyqtSignal(str)
    cl1 = pyqtSignal(str)
    clc1 = pyqtSignal(str)
    alm1 = pyqtSignal(int)

    def __init__(self):
        super(settings, self).__init__()
        self.resize(300, 300)
        self.cwd = os.getcwd()

        self.setWindowTitle('Settings')

        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)

        title1 = QLabel('Yolov3 input parameters: ')
        title2 = QLabel('New Functions: ')
        title3 = QLabel('Choosing the your own model: ')

        title1.setStyleSheet("font-weight:bold;font-size:16px;")
        title2.setStyleSheet("font-weight:bold;font-size:16px;")
        title3.setStyleSheet("font-weight:bold;font-size:16px;")

        si = QLabel('Size(n,n) n: ')
        iou = QLabel("IoU: ")
        score = QLabel('class score: ')

        self.si_in = QLineEdit("544")
        self.iou_in = QLineEdit('0.45')
        self.sc_in = QLineEdit('0.5')

        intValidator = QIntValidator(self)
        intValidator.setRange(224, 992)

        floatleValidator = QDoubleValidator(self)
        floatleValidator.setRange(0, 1)
        floatleValidator.setNotation(QDoubleValidator.StandardNotation)
        floatleValidator.setDecimals(2)

        self.si_in.setValidator(intValidator)
        self.iou_in.setValidator(floatleValidator)
        self.sc_in.setValidator(floatleValidator)

        self.nms = QCheckBox('Double nms', self)
        self.tr = QCheckBox('Tracking',self)

        self.model = QPushButton('Model .h5',self)
        self.path_m = ''

        self.target = QPushButton('Classes .txt',self)
        self.path_t = ''

        self.model.clicked.connect(self.click_b_mo)
        self.target.clicked.connect(self.click_b_ta)

        t = QLabel('The class you want to make warning: ')
        self.cls = QLineEdit("Head")

        self.alm = QCheckBox('Open alarm', self)

        okButton = QPushButton("OK")
        cancelButton = QPushButton("Cancel")
        okButton.clicked.connect(self.click_b_ok)
        cancelButton.clicked.connect(self.close)

        # sub layout
        hbox1 = QHBoxLayout()
        hbox1.addWidget(si)
        hbox1.addWidget(self.si_in)
        hbox1.addWidget(iou)
        hbox1.addWidget(self.iou_in)
        hbox1.addWidget(score)
        hbox1.addWidget(self.sc_in)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.nms)
        hbox2.addWidget(self.tr)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.model)
        hbox3.addWidget(self.target)

        hbox4 = QHBoxLayout()
        hbox4.addWidget(t)
        hbox4.addWidget(self.cls)
        hbox4.addStretch(1)

        hbox5 = QHBoxLayout()
        hbox5.addStretch(1)
        hbox5.addWidget(okButton)
        hbox5.addWidget(cancelButton)

        # main layout
        vbox = QVBoxLayout()

        vbox.addWidget(title1)
        vbox.addLayout(hbox1)

        vbox.addWidget(title2)
        vbox.addLayout(hbox2)

        vbox.addWidget(title3)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addWidget(self.alm)

        vbox.addStretch(1)
        vbox.addLayout(hbox5)
        vbox.setSpacing(20)

        self.setLayout(vbox)

    def click_b_mo(self):
        Name, Type = QFileDialog.getOpenFileName(self, "Choose model .h5", self.cwd + '/model_data',
                                                       "*.h5;;")

        if Name == "":
            return
        else:
            self.path_m = Name

    def click_b_ta(self):
        Name, Type = QFileDialog.getOpenFileName(self, "Choose classes file", self.cwd + '/model_data',
                                                       "*.txt;;")

        if Name == "":
            return
        else:
            self.path_t = Name

    def click_b_ok(self):

        if self.path_m != '' and self.path_t == '':
            reply = QtWidgets.QMessageBox.question(self, 'Warning', "Did not choose model file",
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        elif self.path_m == '' and self.path_t != '':
            reply = QtWidgets.QMessageBox.question(self, 'Warning', "Did not choose class file",
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                   QtWidgets.QMessageBox.No)
        else:
            self.mo1.emit(self.path_m)
            self.cl1.emit(self.path_t)

            si = self.si_in.text()
            io = self.iou_in.text()
            sc = self.sc_in.text()
            cls = self.cls.text()

            self.si1.emit(si)
            self.io1.emit(io)
            self.sc1.emit(sc)

            if self.nms.isChecked():
                self.nm1.emit(1)
            else:
                self.nm1.emit(0)

            if self.tr.isChecked():
                self.tr1.emit(1)
            else:
                self.tr1.emit(0)

            if self.alm.isChecked():
                self.alm1.emit(1)
            else:
                self.alm1.emit(0)

            self.clc1.emit(cls)


            self.close()


# Main window
class GUI(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.initGUI()

        self.video = 1
        self.oc = 1
        self.filtered_class = 'Head'

        # detected times
        self.d_num = 0
        self.und_num = 0

        # new functions
        self.nms = 0
        self.tk = 0
        self.alm = 0

        self.model_path = 'model_data/yolov3-SH-new.h5'  # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/voc_classes.txt'

        self.score = 0.5
        self.iou = 0.45
        self.model_image_size = (544, 544)  # fixed size or (None, None), hw
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def initGUI(self):
        self.resize(1450, 900)
        self.cwd = os.getcwd()

        # window centre
        qr = self.frameGeometry()
        cp = QtWidgets.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        # Title
        self.setWindowTitle('YOLOv3')

        # Picture and video window
        self.label_t = QLabel(self)
        self.label_t.setFixedSize(950, 50)
        self.label_t.move(200, 25)
        self.label_t.setText("Image window")
        self.label_t.setAlignment(Qt.AlignCenter)
        self.label_t.setStyleSheet("color:rgb(10,10,10,255);font-size:45px;font-weight:bold;font-family:Roman times;")

        self.label = QLabel(self)
        self.label.setFixedSize(950, 650)
        self.label.move(200, 100)

        self.label.setStyleSheet("background:white;")

        # information - Warning
        self.war_t = QLabel(self)
        self.war_t.setFixedSize(200, 50)
        self.war_t.move(1200, 25)
        self.war_t.setText("Warning list")
        self.war_t.setAlignment(Qt.AlignCenter)
        self.war_t.setStyleSheet("color:rgb(10,10,10,255);font-size:20px;font-weight:bold;font-family:Roman times;")

        self.war = QtWidgets.QLabel(self)
        self.war.move(1200, 100)
        self.war.setFixedSize(200, 650)
        self.war.setText(' ')
        self.war.setObjectName('label')
        self.war.setAlignment(Qt.AlignCenter)

        self.war.setStyleSheet("color: rgb(255, 0, 0); background:white;font-size:18px;")


        # button - settings
        self.b_set = QtWidgets.QPushButton('Settings', self)
        self.b_set.setGeometry(50, 100, 100, 50)

        self.b_set.clicked.connect(self.click_b_set)

        # button - open picture and detect
        self.b_pic = QtWidgets.QPushButton('Picture', self)
        self.b_pic.setGeometry(50, 250, 100, 50)

        self.b_pic.clicked.connect(self.click_b_pic)

        # button - open video and detect
        self.b_vid = QtWidgets.QPushButton('Video', self)
        self.b_vid.setGeometry(50, 400, 100, 50)

        self.b_vid.clicked.connect(self.click_b_vid)

        # button - open camera and detect
        self.b_cam = QtWidgets.QPushButton('Camera', self)
        self.b_cam.setGeometry(50, 550, 100, 50)

        self.b_cam.clicked.connect(self.click_b_cam)

        # button - Default settings
        self.b_d = QtWidgets.QPushButton('Default', self)
        self.b_d.setGeometry(50, 700, 100, 50)

        self.b_d.clicked.connect(self.click_b_d)

        # window show
        self.show();


    #-------------- detecting functions---------------------

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))
        print("Class path: " + self.classes_path)
        print("Model path: " + self.model_path)
        print(self.model_image_size)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))

        # Original Version
        if self.nms == 0:
            print('Use original version')
            print("IoU: %f", self.iou)
            print("Class Score: %f", self.score)
            boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                                    len(self.class_names), self.input_image_shape,
                                                    score_threshold=self.score, iou_threshold=self.iou)
        # Modified version of double NMS
        else:
            print('Use double NMS')
            print("IoU: %f", self.iou)
            print("Class Score: %f", self.score)
            boxes, scores, classes = yolo_eval_mine2(self.yolo_model.output, self.anchors,
                                                    len(self.class_names), self.input_image_shape,
                                                    score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def detect_image(self, image):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')

        # print(" image_data.shape:",image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        start = timer()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 600

        end = timer()

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        d_signal = 0
        return_boxs = []
        class_n = []
        for i, c in reversed(list(enumerate(out_classes))):

            box = out_boxes[i]
            score = out_scores[i]

            # ----------------------boxes for tracking----------------------------

            predicted_class = self.class_names[c]
            # if predicted_class != 'Head':
            # continue
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxs.append([x, y, w, h])
            class_n.append(predicted_class)

            # --------------------------------------------------------------------

            if predicted_class == self.filtered_class:
                d_signal = 1

            label = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            center_y = (top + bottom) / 2
            center_x = (left + right) / 2
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
                draw.point((center_x, center_y), fill=(255, 0, 0))
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        # count the continuous detected time of the target
        if d_signal == 1:
            self.d_num += 1
            self.und_num = 0
        else:
            self.und_num += 1
            self.d_num = 0
        return image, return_boxs, class_n


    def detect_video(self, vid):

        fps = 0
        while self.oc == 1:
            t1 = time.time()

            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            r_image, b, c = self.detect_image(image)
            if self.d_num == 3:
                self.alarm(self.filtered_class + ' detected!')
            elif self.und_num == 5:
                self.war.setText(' ')

            fps = (fps + (1. / (time.time() - t1))) / 2
            fps1 = round(fps, 2)

            img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
            cv2.putText(img, "FPS {0}".format(str(fps1)), (10, 40), 2, 1, (0, 0, 255), 3)
            img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            img = img.scaled(950, 650)
            self.label.setPixmap(QtGui.QPixmap.fromImage(img))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()





    # --------------tracking functions---------------------

    def compute_iou(self, rec1, rec2):
        # computing area of each rectangles
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])

        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0

        else:
            S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
            return S_cross / (S1 + S2 - S_cross)


    def tracking(self, vid):

        # Definition of the parameters
        max_cosine_distance = 0.3
        nn_budget = None

        # deep_sort
        model_filename = 'model_data/mars-small128.pb'  # appearance feature generated by CNN.
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        fps = 0
        while self.oc == 1:
            ret, frame = vid.read()
            if ret != True:
                break;
            t1 = time.time()

            image = Image.fromarray(frame)

            time3 = time.time()
            ig, boxs, cl = self.detect_image(image)
            ig = cv2.cvtColor(np.asarray(ig), cv2.COLOR_RGB2BGR)
            time4 = time.time()

            #print('detect cost is', time4 - time3)
            #print("box_num", len(boxs))
            time3 = time.time()
            features = encoder(frame, boxs)

            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

            time4 = time.time()
            #print('features extract is', time4 - time3)

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            b = []
            for i in range (len(cl)):
                if cl[i] == self.filtered_class and i<len(detections):
                    b.append(detections[i].to_tlbr())

            nu = 0
            id = []
            for track in tracker.tracks:
                #track = tracker.tracks[i]
                if track.is_confirmed() and track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()

                for l in b:
                    a = self.compute_iou(l, bbox)
                    #print(a)
                    if a >= 0.8:

                        cv2.rectangle(ig, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255),
                                      2)
                        cv2.putText(ig, str(track.track_id)+' '+self.filtered_class, (int(bbox[2]), int(bbox[3])), 0, 5e-3 * 200,
                                    (0, 255, 0), 2)
                        id.append(track.track_id)
                        #b.remove(l)
                        break
                        nu = 1

                if nu == 0:
                    cv2.rectangle(ig, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255),
                                  2)
                    cv2.putText(ig, str(track.track_id), (int(bbox[2]), int(bbox[3])), 0, 5e-3 * 200,
                                (0, 255, 0), 2)

            # make alarm according to the id
            trs = ''
            for i in id:
                trs = trs+'For number '+str(i)+" detected " + self.filtered_class+"\n"

            self.alarm(trs)

            fps = (fps + (1. / (time.time() - t1))) / 2
            fps1 = round(fps, 2)
            cv2.putText(ig, "FPS {0}".format(str(fps1)), (10, 40), 2, 1, (0, 0, 255), 3)

            img = ig
            img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            img = img.scaled(950, 650)
            self.label.setPixmap(QtGui.QPixmap.fromImage(img))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()




    #-------------- buttons---------------------------------

    def click_b_pic(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "Open picture", self.cwd + '/test', "*.jpg;;*.png;;*.jpeg;;All Files (*)")

        if imgName == "":
            return

        image = Image.open(imgName)
        r_image, bs, c = self.detect_image(image)
        self.result = np.asarray(r_image)
        self.result = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (950, 650), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))


    def click_b_vid(self):

        if self.video==1:

            fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                    "Open video",
                                                                    self.cwd + '/test',
                                                                    "*.mov;;*.mp4;;All Files (*);;")

            if fileName_choose == "":
                return

            vid = cv2.VideoCapture(fileName_choose)
            if not vid.isOpened():
                QtWidgets.QMessageBox.question(self, "Video detection", 'Error: ' + "Couldn't open webcam or video",
                                               QtWidgets.QMessageBox.Ok,
                                               QtWidgets.QMessageBox.Ok)

            else:
                self.b_vid.setText('Close video')
                self.video = 0
                self.oc = 1

                if self.tk == 0:
                    self.detect_video(vid)
                else:
                    self.tracking(vid)

        else:
            self.b_vid.setText('Video')
            self.oc=0
            self.video=1
            self.label.clear()
            self.war.setText(' ')


    def click_b_cam(self):

        value, ok = QInputDialog.getText(self, "Input IP address", 'rtsp://[username]:[passwd]@[ip]:[port]/[codec]/[channel]/[subtype]/av_stream',QLineEdit.Normal, "dasada")

        if ok :
            if self.video == 1:

                cap = cv2.VideoCapture(value)
                if not cap.isOpened():
                    QtWidgets.QMessageBox.question(self, "Surveillance camera detection",
                                                   'Error: ' + "Couldn't open camera", QtWidgets.QMessageBox.Ok,
                                                   QtWidgets.QMessageBox.Ok)

                else:
                    self.b_cam.setText('Close camera')
                    self.video = 0
                    self.oc = 1

                    if self.tk == 0:
                        self.detect_video(cap)
                    else:
                        self.tracking(cap)

            else:
                self.b_cam.setText('Video')
                self.oc = 0
                self.video = 1
                self.label.clear()
                self.war.setText(' ')


    def click_b_set(self):

        self.s = settings()
        self.s.show()
        self.s.si1.connect(self.getsi)
        self.s.io1.connect(self.getiou)
        self.s.sc1.connect(self.getsc)
        self.s.nm1.connect(self.getnms)
        self.s.tr1.connect(self.gettra)
        self.s.mo1.connect(self.getmo)
        self.s.cl1.connect(self.getcl)
        self.s.clc1.connect(self.getclc)
        self.s.alm1.connect(self.getalm)


    def click_b_d(self):
        self.filtered_class = 'Head'
        self.nms = 0
        self.tk = 0
        self.classes_path = 'model_data/voc_classes.txt'
        self.model_path = 'model_data/yolov3-SH-new.h5'
        self.score = 0.5
        self.iou = 0.45
        self.model_image_size = (544, 544)
        self.alm = 0
        self.boxes, self.scores, self.classes = self.generate()
        QtWidgets.QMessageBox.question(self, "Settings",
                                       'Success: ' + "Change to default settings.", QtWidgets.QMessageBox.Ok,
                                       QtWidgets.QMessageBox.Ok)



    # -------------- receiving settings----------------------

    def getsi(self, parameter):
        if parameter != '':
            self.model_image_size = (int(parameter), int(parameter))

    def getiou(self, parameter):
        if parameter != '':
            self.iou = float(parameter)

    def getsc(self, parameter):
        if parameter != '':
            self.score = float(parameter)

    def getnms(self, parameter):
        self.nms = int(parameter)

    def gettra(self, parameter):
        self.tk = parameter

    def getmo(self, parameter):
        if parameter != '':
            self.model_path = parameter

    def getcl(self, parameter):
        if parameter != '':
            self.classes_path = parameter

    def getalm(self, parameter):
        self.alm = parameter

    def getclc(self, parameter):
        if parameter != '':
            self.filtered_class = parameter
        self.boxes, self.scores, self.classes = self.generate()



    # ------------------------alarm-----------------------------------
    def alarm(self, inf):
        sys = platform.system()
        if sys == "Windows":

            self.war.setText(inf)
            duration = 3000  # millisecond
            freq = 440  # Hz
            if self.alm == 0:
                pass
            else:
                winsound.Beep(freq, duration)
        elif sys == "Linux":

            print("Alarm not supported for Linux!!!")
            pass
        else:
            pass


    def closeEvent(self, QCloseEvent):

        reply = QtWidgets.QMessageBox.question(self, 'Warning',"Sure to close the window?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            QCloseEvent.accept()

        else:
            QCloseEvent.ignore()


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())