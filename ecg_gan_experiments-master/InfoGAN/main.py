__author__ = "Sereda"
import os
import json
from InfoGAN.train import train, Params



name = "E_256len_qrs_5steps_4_ecgs_homogene_short "
os.makedirs(name, exist_ok=True)
parameters = Params(name)

with open(name+"/params.json", "w") as json_file:
    json.dump(parameters._asdict(), json_file)
train(parameters, selected_ecgs=['1102593395', '50628511', '1102527952', '50501116'])





