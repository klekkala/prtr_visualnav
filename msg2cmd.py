import rospy
from sensor_msgs.msg import CompressedImage
from carla_navigation.msg import TimedTwist
import signal
import sys
import numpy as np
import random
import torch
import tensorflow as tf
from cv_bridge import CvBridge
import cv2
from models import VAEBEV, MDLSTM
import matplotlib.pyplot as plot
import rosbag
import time
from fpvbev import Encoder
from sklearn.metrics import mean_squared_error
use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

files = [
{"name":"test_2023-08-20-01-42-19", "len":1316, 0:1, 542:4, 630:5, 659:1, 768:2, 785:1, "end":2692520463},
{"name":"test_2023-08-20-01-32-33", "len":2558, 0:3, 10:1, 84:2, 111:1, 400:2, 407:5, 442:2, 452:1, 575:3, 633:1, 734:3, 770:1, 841:2, 845:1, 919:3, 925:1, 962:2, 969:0, 993:2, 998:1, 1040:3, 1062:1, 1223:2, 1237:1,  1145:2, 1148:1, 1251:5, 1331:1, 1348:2, 1354:0, 1365:1, 1386:5, 1448:1, "end":1692520463},
{"name":"test_2023-08-20-01-25-41", "len":2555, 0:1, 668:0, 769:3, 813:0, 879:2, 922:1, 1109:3, 1187:1, 2207:3, 2224:0, 2236:2, 2241:1, 2268:3, 2287:0, 2303:2, 2309:1, 2322:3, "end":2692520463},
{"name":"test_2023-08-20-01-52-47", "len":2076, 0:1, 146:3, 168:1, 260:3, 332:1, 464:3, 531:0, 417:3, 473:1, 660:2, 704:0, 754:2, 861:1, 1855:3, 1903:1, "end":2692520463},

          ]
'''
{"name":"test_2023-08-20-01-13-10", "len":1568, 0:0, 56:3, 61:1},
{"name":"test_2023-08-20-01-16-18", "len":1641, 0:1, 573:3, 583:1, 626:3, 665:1, 1132:2, 1193:1, 1262:3, 1333:1},
{"name":"test_2023-08-20-01-18-45", "len":1333, 0:1, 825:3, 864:1, 925:3, 979:1},
{"name":"test_2023-08-20-01-22-05", "len":3130, 0:1, 163:4, 303:1, 760:3, 792:0, 834:3, 837:1, 2016:3, 2030:0, 2066:3, 2072:1, 2110:3, 2130:1},
{"name":"test_2023-08-20-01-28-36", "len":3489, 0:0, 772:3, 779:1, 917:3, 927:0, 951:3, 958:1, 1382:4, 1476:1, 1729:2, 1735:1, 1037:2, 2047:0, 2091:2, 2097:1, 3104:4, 3175:5, 3256:1},
{"name":"test_2023-08-20-01-32-33", "len":2558, 0:1, 557:2, 595:1, 734:3, 780:1, 1115:2, 1124:0, 1145:2, 1148:1, 1503:2, 1524:1, 2020:3, 2036:0, 2079:3, 2083:1},
{"name":"test_2023-08-20-01-35-35", "len":1230, 0:3, 50:0, 60:2, 67:1, 958:4, 1058:5, 1087:4, 1178:5},
{"name":"test_2023-08-20-01-38-32", "len":3234, 0:3, 64:1, 293:3, 346:5, 634:4, 678:1, 1048:3, 1056:0, 1085:3, 1099:1, 1143:2, 1154:0, 1189:2, 1192:1},
{"name":"test_2023-08-20-01-43-52", "len":3442, 0:2, 63:1, 1144:4, 1309:5, 1332:1, 2728:3, 2748:1, 2844:4, 3128:1},
{"name":"test_2023-08-20-01-47-48", "len":1738, 0:0, 72:3, 79:1, 1227:4, 1362:5, 1448:1, 1498:5, 1523:1},
{"name":"test_2023-08-20-01-49-47", "len":2638, 0:1, 2616:2, 2029:0},
{"name":"test_2023-08-20-01-52-47", "len":2076, 0:0, 46:2, 53:1, 417:3, 473:1, 699:4, 821:5, 848:4, 944:2, 947:1},
{"name":"test_2023-08-20-01-55-13", "len":1910, 0:1, 309:4, 473:5, 506:1, 1221:3, 1230:0, 1277:3, 1282:1, 1602:4, 1659:5, 1712:1},
{"name":"test_2023-08-20-01-58-36", "len":1538, 0:1, 315:3, 323:0, 347:3, 350:1, 1161:0, 1178:3, 1245:0, 1276:3, 1310:1},
'''

vae_model_path = "/home2/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
naive_lstm_path = "/home2/ckpts/pretrained/carla/BEV_LSTM_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_1_512.pt"
cnn_model_path = "/home/carla/img2cmd_data/model/cnn"


def cross_entropy(y_true, y_pred):
    loss = 0
    for i in range(len(y_pred)):
        if y_pred[i] <= 0:
            y_pred[i] += 0.001
        loss = loss + (-1 * y_true[i] * np.log(y_pred[i]))

    return loss

def image_callback(img):
    global img_msg
    img_msg = img

def method(action, out):
    global img_msg
    global i
    global z
    global window
    global correction
    global actions
    global prediction

    if abs(action.twist.angular.z) < 0.4 and  abs(action.twist.angular.z) > 0.1:
        action.twist.angular.z *= 2
    a = [np.clip(action.twist.linear.x, 0, 1), np.clip(round(-action.twist.angular.z/3, 2), -1, 1)]
    actions.append(action)
    s = None
    if img_msg is not None:
        for action in actions:
            if abs(img_msg.header.stamp.to_sec() - action.header.stamp.to_sec()-delay) < 0.03:
                i += 1
                actions = actions[1:]
                cos = torch.nn.CosineSimilarity(dim=0)

                np_arr = np.frombuffer(img_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                #cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                RGB_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                #RGB_img = RGB_img[:, 280:-280]
                RGB_img = cv2.resize(RGB_img, (84, 84), interpolation=cv2.INTER_LINEAR)

                rid, score, image_embed = encoder(RGB_img)  # FPV embedding and approx

                if i < 11:
                    z = torch.unsqueeze(encoder.anchors_lr[rid], dim=0) # input to lstm
                    r__ = r_ = torch.reshape(bev_lstm.vae.recon(image_embed),  # FPV-BEV
                                       (64, 64)).cpu().numpy() * 255
                    w = 1
                    window.append(0)

                    break
                r_ = torch.reshape(bev_lstm.vae.recon(image_embed),  # FPV-BEV
                                   (64, 64)).cpu().numpy() * 255
                r__ = torch.reshape(bev_lstm.vae.recon(z),           # LSTM raw output
                                    (64, 64)).cpu().numpy() * 255

                mse = mean_squared_error(r_.reshape((1, 64 * 64)),
                                         r__.reshape((1, 64 * 64)))
                if score > 0.9:
                    window = window[1:]
                    window.append(1 if mse > 9000 else 0)
                    w = 1
                else:
                    w = 0

                z = image_embed*w + z*(1-w)
                id, _, _ = encoder(z, False)
                s = torch.unsqueeze(encoder.anchors_lr[id], dim=0) # input to lstm
                r_ = cv2.resize(r_, (200, 200), interpolation=cv2.INTER_LINEAR)
                r__ = cv2.resize(r__, (200, 200), interpolation=cv2.INTER_LINEAR)
                cv_image[200:200 + 200, 100:100 + 200] = cv2.cvtColor(r_, cv2.COLOR_GRAY2RGB)
                cv_image[450:450 + 200, 100:100 + 200] = cv2.cvtColor(r__, cv2.COLOR_GRAY2RGB)
                #Qcv2.putText(cv_image, "mse: " + str(mse), (700, 650), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #            (255, 0, 0), 3)
                #cv2.putText(cv_image, "score: " + str(round(score, 2)), (200, 250),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                #cv2.imwrite("/home2/carla/2023_08_20/" + fn + "/" + str(i) + ".jpg", cv_image)
                out.write(cv_image)
                print(i)
                img_msg = None

                break
            elif img_msg.header.stamp.to_sec() - action.header.stamp.to_sec() - delay > 0.03:
                actions = actions[1:]
            elif img_msg.header.stamp.to_sec() - action.header.stamp.to_sec() - delay < -0.03:
                break

    if z is not None:
        out = bev_lstm(torch.Tensor([a]).to(device), z)
        mus = out[0][0]
        pi = torch.exp(out[2][0])
        z = (mus[0]*pi[0]+mus[1]*pi[1]+mus[2]*pi[2]+mus[3]*pi[3]+mus[4]*pi[4]).unsqueeze(0)


if __name__ == '__main__':
    vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    bev_lstm = MDLSTM(latent_size=32, action_size=2, hidden_size=256, num_layers=1, gaussian_size=5,
                      vae=vae).to(device)
    bev_lstm.eval()
    bev_lstm.init_hs()
    checkpoint = torch.load(naive_lstm_path, map_location="cpu")

    bev_lstm.load_state_dict(checkpoint['model_state_dict'])
    for param in bev_lstm.parameters():
        param.requires_grad = False

    vae_ckpt = torch.load(vae_model_path, map_location="cpu")
    bev_lstm.vae.load_state_dict(vae_ckpt['model_state_dict'])

    delay = 10

    anchors = []
    for i in range(6):
        im = cv2.imread("train/"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE).astype(float)
        anchors.append(im)

    cnn = tf.keras.models.load_model(cnn_model_path)
    encoder = Encoder("/home2/ckpts/pretrained/models/FPV_BEV_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_128_512.pt", False)

    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter("lstm.mp4", fourcc, 100.0, (1280, 720))
    bridge = CvBridge()
    for f in files:
        # read bag file
        print("%", f["name"])
        fn = f["name"]
        bag = rosbag.Bag('/home2/carla/2023_08_20/cam1/' + f["name"] + ".bag")
        labels = [-1] * f["len"]
        prev_k = 0
        prev_l = f[0]
        for k in f.keys():
            if isinstance(k, int) and k > 0:
                labels[prev_k:k] = [prev_l] * (k - prev_k)
                prev_k = k
                prev_l = f[k]
        labels[prev_k:f["len"]] = [prev_l] * (f["len"] - prev_k)





        # initialize
        loss = {"mse": {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                "entropy": {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                "accuracy": {0: [], 1: [], 2: [], 3: [], 4: [], 5: []},
                }
        i = 0
        img_msg = None
        z = None
        window = []
        actions = []
        correction = False
        img_id = -1
        prediction = -1
        obs = {"RGB":None, "id":None, "score":None, "embed":None}

        # run
        prev_t = -1
        for topic, msg, t in bag.read_messages(topics=['/cam1/color/image_raw/compressed', '/cmd_vel']):
            if int(t.to_sec()) > f["end"]:
                break
            if topic == "/cam1/color/image_raw/compressed":
                msg.header.stamp = t
                image_callback(msg)
            elif topic == "/cmd_vel":
                timed_msg = TimedTwist()
                timed_msg.header.stamp = t
                timed_msg.twist = msg
                method(timed_msg, out)
            if prev_t > 0:
                time.sleep((t.to_nsec() - prev_t) / 1000000000.0)
            prev_t = t.to_nsec()

        out.release()
        cap.release()
        bag.close()


