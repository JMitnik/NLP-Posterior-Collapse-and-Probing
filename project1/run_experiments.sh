# 1 experiments are the ones to begin with
python main.py --run_label='-1-epochs5-wdropout1-mu3' --nr_epochs=5 --wdropout_k=1 --mu_beta=3
python main.py --run_label='-1-epochs5-wdropout1-mu5' --nr_epochs=5 --wdropout_k=1 --mu_beta=5
python main.py --run_label='-1-epochs5-wdropout0_5-mu1' --nr_epochs=5 --wdropout_k=0.5 --mu_beta=1
python main.py --run_label='-1-epochs5-wdropout0_5-mu3' --nr_epochs=5 --wdropout_k=0.5 --mu_beta=3
python main.py --run_label='-1-epochs5-wdropout0_5-mu5' --nr_epochs=5 --wdropout_k=0.5 --mu_beta=5

python main.py --run_label='-1-epochs5-wdropout0-mu1' --nr_epochs=5 --wdropout_k=0 --mu_beta=1
python main.py --run_label='-1-epochs5-wdropout0-mu3' --nr_epochs=5 --wdropout_k=0 --mu_beta=3
python main.py --run_label='-1-epochs5-wdropout0-mu5' --nr_epochs=5 --wdropout_k=0 --mu_beta=5
