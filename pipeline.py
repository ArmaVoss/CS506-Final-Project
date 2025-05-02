import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

for i in range(5):
    print("")

print("XGBoostModel")
time.sleep(2) 
subprocess.run([sys.executable, "XGBoost.py"], check=True)

for i in range(5):
    print("")

print("MLP, LogisticReg, DecisionTreeClassifier")
time.sleep(2) 

print("MLP takes like 15-20 Seconds Be Patient For Training")
subprocess.run([sys.executable, "mlp_logistic_decision.py"], check=True)

