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

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

files = [{"name":"test_2023-08-20-01-07-49", "len":834, 0:1, 658:3, 666:1},
{"name":"test_2023-08-20-01-09-08", "len":2028, 0:1, 440:3, 459:1, 584:3, 591:1, 2003:2},
{"name":"test_2023-08-20-01-11-26", "len":1456, 0:2, 42:1, 980:4, 1037:5, 1070:4, 1102:5, 1177:1},
{"name":"test_2023-08-20-01-13-10", "len":1568, 0:0, 56:3, 61:1},
{"name":"test_2023-08-20-01-16-18", "len":1641, 0:1, 573:3, 583:1, 626:3, 665:1, 1132:2, 1193:1, 1262:3, 1333:1},
{"name":"test_2023-08-20-01-18-45", "len":1333, 0:1, 825:3, 864:1, 925:3, 979:1},
{"name":"test_2023-08-20-01-22-05", "len":3130, 0:1, 163:4, 303:1, 760:3, 792:0, 834:3, 837:1, 2016:3, 2030:0, 2066:3, 2072:1, 2110:3, 2130:1},
{"name":"test_2023-08-20-01-25-41", "len":2555, 0:0, 56:2, 59:1, 887:0, 916:3, 922:1, 968:3, 988:0, 1047:2, 1064:1, 2382:3, 2389:0, 2399:2, 2402:1, 2445:3, 2454:0, 2467:2, 2471:1},
{"name":"test_2023-08-20-01-28-36", "len":3489, 0:0, 772:3, 779:1, 917:3, 927:0, 951:3, 958:1, 1382:4, 1476:1, 1729:2, 1735:1, 1037:2, 2047:0, 2091:2, 2097:1, 3104:4, 3175:5, 3256:1},
{"name":"test_2023-08-20-01-32-33", "len":2558, 0:1, 557:2, 595:1, 734:3, 780:1, 1115:2, 1124:0, 1145:2, 1148:1, 1503:2, 1524:1, 2020:3, 2036:0, 2079:3, 2083:1},
{"name":"test_2023-08-20-01-35-35", "len":1230, 0:3, 50:0, 60:2, 67:1, 958:4, 1058:5, 1087:4, 1178:5},
{"name":"test_2023-08-20-01-38-32", "len":3234, 0:3, 64:1, 293:3, 346:5, 634:4, 678:1, 1048:3, 1056:0, 1085:3, 1099:1, 1143:2, 1154:0, 1189:2, 1192:1},
{"name":"test_2023-08-20-01-42-19", "len":1316, 0:3, 18:0, 61:2, 66:1, 587:4, 712:5, 785:1},
{"name":"test_2023-08-20-01-43-52", "len":3442, 0:2, 63:1, 1144:4, 1309:5, 1332:1, 2728:3, 2748:1, 2844:4, 3128:1},
{"name":"test_2023-08-20-01-47-48", "len":1738, 0:0, 72:3, 79:1, 1227:4, 1362:5, 1448:1, 1498:5, 1523:1},
{"name":"test_2023-08-20-01-49-47", "len":2638, 0:1, 2616:2, 2029:0},
{"name":"test_2023-08-20-01-52-47", "len":2076, 0:0, 46:2, 53:1, 417:3, 473:1, 699:4, 821:5, 848:4, 944:2, 947:1},
{"name":"test_2023-08-20-01-55-13", "len":1910, 0:1, 309:4, 473:5, 506:1, 1221:3, 1230:0, 1277:3, 1282:1, 1602:4, 1659:5, 1712:1},
{"name":"test_2023-08-20-01-58-36", "len":1538, 0:1, 315:3, 323:0, 347:3, 350:1, 1161:0, 1178:3, 1245:0, 1276:3, 1310:1},

          ]


vae_model_path = "/lab/kiran/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
naive_lstm_path = "/lab/kiran/ckpts/pretrained/carla/BEV_LSTM_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_1_512.pt"
cnn_model_path = "/lab/kiran/img2cmd_data/model/cnn"


def image_callback(img):
    global img_msg
    global img_id
    img_msg = img
    img_id += 1

def action_callback(action):
    global img_msg
    global i
    global z_in
    global window
    global correction
    global actions
    global prediction
    i += 1

    a = [np.clip(action.twist.linear.x, 0, 1), np.clip(round(-action.twist.angular.z/2, 1), -1, 1)]
    actions.append(action)

    if img_msg is not None:
        for action in actions:
            if abs(img_msg.header.stamp.to_sec() - action.header.stamp.to_sec()-delay) < 0.03:
                actions = actions[1:]
                cos = torch.nn.CosineSimilarity(dim=0)

                np_arr = np.fromstring(img_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                #cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                RGB_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                RGB_img = cv2.resize(RGB_img, (84, 84), interpolation=cv2.INTER_LINEAR)
                RGB_img = np.expand_dims(RGB_img, axis=0) / 255.0
                probs = cnn.predict(RGB_img, verbose=0)
                id = np.argmax(probs)

                z_obs = bev_lstm.vae.reparameterize(anchors_lr[id][:32], anchors_lr[id][32:])
                z_in = torch.unsqueeze(z_obs, dim=0)
                prediction = id

                r_ = torch.reshape(bev_lstm.vae.recon(z_in), (1, 64 * 64))
                gt = labels[img_id]
                mse = torch.nn.functional.mse_loss(anchors[gt].reshape((1, 64*64)), r_).cpu().numpy()
                entropy = torch.nn.functional.cross_entropy(anchors[gt].reshape((1, 64*64)), r_).cpu().numpy()

                loss["mse"][gt].append(mse)
                loss["entropy"][gt].append(entropy)
                loss["accuracy"][gt].append(1 if prediction == gt else 0)

                img_msg = None

                break
            elif img_msg.header.stamp.to_sec() - action.header.stamp.to_sec() - delay > 0.03:
                actions = actions[1:]
            elif img_msg.header.stamp.to_sec() - action.header.stamp.to_sec() - delay < -0.03:
                break



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

    latent_cls = []   # contains representations of 10 bev classes
    div_val = 255.0
    delay = 1.0

    anchors = []
    anchors_lr = []
    for i in range(6):
        im = cv2.imread("train/"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE)
        with torch.no_grad():
            im = np.expand_dims(im, axis=(0, 1))
            im = torch.tensor(im).to(device) / 255.0
            anchors.append(im)
            _, mu, logvar = bev_lstm.vae(im.float())
            fs = torch.concat((mu, logvar), axis=-1).reshape(64, )
            anchors_lr.append(fs)

    cnn = tf.keras.models.load_model(cnn_model_path)
#pub = rospy.Publisher('cmd', Int16, queue_size=10)

    bridge = CvBridge()
    for f in files:
        print(f["name"])
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

        loss = {"mse": {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []},
                "entropy": {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []},
                "accuracy": {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []},
                }
        i = 0
        img_msg = None
        z_in = None
        window = []
        actions = []
        correction = False
        img_id = -1
        prediction = -1

        prev_t = -1
        for topic, msg, t in bag.read_messages(topics=['/cam1/color/image_raw/compressed', '/cmd_vel']):
            if topic == "/cam1/color/image_raw/compressed":
                msg.header.stamp = t
                image_callback(msg)
            elif topic == "/cmd_vel":
                timed_msg = TimedTwist()
                timed_msg.header.stamp = t
                timed_msg.twist = msg
                action_callback(timed_msg)
            if prev_t > 0:
                time.sleep((t.to_nsec() - prev_t) / 1000000000.0)
            prev_t = t.to_nsec()
        print("mse:")
        for k in sorted(loss["mse"].keys()):
            if loss["mse"][k]:
                print(k, len(loss["mse"][k]), sum(loss["mse"][k]) / len(loss["mse"][k]))
        print("entropy:")
        for k in sorted(loss["entropy"].keys()):
            if loss["entropy"][k]:
                print(k, len(loss["entropy"][k]), sum(loss["entropy"][k]) / len(loss["entropy"][k]))
        print("accuracy:")
        for k in sorted(loss["accuracy"].keys()):
            if loss["accuracy"][k]:
                print(k, len(loss["accuracy"][k]), sum(loss["accuracy"][k]) / len(loss["accuracy"][k]))

    bag.close()


