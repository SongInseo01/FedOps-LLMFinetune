import hydra
from omegaconf import DictConfig

from server_app import FLServer
import models
import data_preparation
from hydra.utils import instantiate



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    """
    Set the initial global model you created in models.py.
    """
    # Build init global model using transformers
    model = None
    model_type = cfg.model_type # Check tensorflow or torch model
    model_name = type(model).__name__
    
    # Start fl server
    fl_server = FLServer(cfg=cfg, model=model, model_name=model_name, model_type=model_type)
    fl_server.start()
    

if __name__ == "__main__":
    main()

