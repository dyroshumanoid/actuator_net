from utils import train_actuator_network_and_plot_predictions

EXPERIMENT_DIR     = '/home/dyros/scraps/actuator_net/data/pkl'
ACTUATOR_NET_PATH  = 'hashi.pt'
DATALOADER_PATH    = 'hashi.dataloader'
MODEL_TYPE         = 'mlp'

train_actuator_network_and_plot_predictions(
    experiment_dir=EXPERIMENT_DIR,
    actuator_network_path=ACTUATOR_NET_PATH,
    dataloader_path=DATALOADER_PATH,
    model_type=MODEL_TYPE,
    load_pretrained_model=False,
)
