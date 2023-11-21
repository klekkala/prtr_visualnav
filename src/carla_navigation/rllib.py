import rospy
from sensor_msgs.msg import CompressedImage
import cv2
from fpvbev import Encoder
from models import VAEBEV, MDLSTM
import torch
import numpy as np
from std_msgs.msg import Int64, Float64
from geometry_msgs.msg import Twist
from ray.rllib.policy.policy import Policy
use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

class Map:
    def __init__(self):
        rospy.init_node('map', anonymous=True)

        vae_model_path = "/home/administrator/ckpts/pretrained/carla/BEV_VAE_CARLA_RANDOM_BEV_CARLA_STANDARD_0.01_0.01_256_64.pt"
        naive_lstm_path = "/home/administrator/ckpts/pretrained/carla/BEV_LSTM_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_1_512.pt"

        vae = VAEBEV(channel_in=1, ch=16, z=32).to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False

        self.bev_lstm = MDLSTM(latent_size=32, action_size=2, hidden_size=256, num_layers=1, gaussian_size=5,
                          vae=vae).to(device)
        self.bev_lstm.eval()
        self.bev_lstm.init_hs()
        checkpoint = torch.load(naive_lstm_path, map_location="cpu")

        self.bev_lstm.load_state_dict(checkpoint['model_state_dict'])
        for param in self.bev_lstm.parameters():
            param.requires_grad = False

        vae_ckpt = torch.load(vae_model_path, map_location="cpu")
        self.bev_lstm.vae.load_state_dict(vae_ckpt['model_state_dict'])

        self.encoder = Encoder("/home/administrator/ckpts/pretrained/carla/FPV_BEV_CARLA_RANDOM_BEV_CARLA_STANDARD_0.1_0.01_128_512.pt",
                          False)

        self.aux = 0
        self.i = 0
        self.window = []
        self.z = None
        self.rgb = None
        self.state = my_restored_policy.get_initial_state()
        
        
        self.map_pub = rospy.Publisher(
            '/occ_map', Int64, queue_size=1)

        self.aux_pub = rospy.Publisher(
            '/aux', Float64, queue_size=1)
            
        self.action_pub = rospy.Publisher(
            '/agent_vel', Twist, queue_size=1)

        self.img_sub = rospy.Subscriber(
            '/cam1/color/image_raw/compressed', CompressedImage, self.img_cb)

        self.action_sub = rospy.Subscriber(
            '/cmd_vel', Twist, self.method)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()
    
    def rgb_cb(self, rgb):
        np_arr = np.frombuffer(rgb.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        RGB_img = cv2.resize(cv_image, (84, 84), interpolation=cv2.INTER_LINEAR)   
        
        self.rgb = RGB_img
        print(1)

    def method(self, action):
        self.i += 1
        if self.rgb is None:
            if self.z is None:
                return
            id, _, _ = self.encoder(self.z, False)
            s = self.z
        else:
            rid, score, image_embed = self.encoder(self.rgb)  # FPV embedding and approx
            self.rgb = None

            if self.i < 20:
                id = rid
                s = image_embed  # input to lstm

                r_ = r__ = torch.reshape(self.bev_lstm.vae.recon(image_embed),  # FPV-BEV
                           (64, 64)).cpu().numpy() * 255
                self.window.append(0)

            else:
                # o = encoder.anchors[nid].reshape((64, 64))  # BEV output at t
                r_ = torch.reshape(self.bev_lstm.vae.recon(image_embed),  # FPV-BEV
                           (64, 64)).cpu().numpy() * 255
                r__ = torch.reshape(self.bev_lstm.vae.recon(self.z),  # LSTM raw output
                            (64, 64)).cpu().numpy() * 255

                mse = mean_squared_error(r_.reshape((1, 64 * 64)),
                                 r__.reshape((1, 64 * 64)))

                if score > 0.8:
                    self.window = self.window[1:]
                    self.window.append(1 if mse > 9000 else 0)
                    w = sum(self.window) / len(self.window)
                else:
                    w = 0

                self.z = image_embed * w + self.z * (1 - w)
                id, _, _ = self.encoder(self.z, False)
                s = self.z #torch.unsqueeze(encoder.anchors_lr[id], dim=0)

        a = [action.linear.x, action.angular.z]
        obs = {"obs":self.encoder.anchors[id], "aux":self.aux}

        out = self.bev_lstm(torch.Tensor([a]).to(device), s)
        mus = out[0][0]
        pi = torch.exp(out[2][0])
        self.z = (mus[0] * pi[0] + mus[1] * pi[1] + mus[2] * pi[2] + mus[3] * pi[3] + mus[4] * pi[4]).unsqueeze(0)

        a, self.state, _ = my_restored_policy.compute_single_action(obs, self.state, prev_action=a, prev_reward=0)

        action.linear.x = a[0]
        action.angular.z = a[1]
        self.action_pub.publish(action)
        self.map_pub.publish(id)

        cv2.imshow("1", self.encoder.anchors[id])
        cv2.waitKey(1)



if __name__ == '__main__':
    model_path = "/home/administrator/checkpoint"
    my_restored_policy = Policy.from_checkpoint(model_path)
    map_ = Map()
