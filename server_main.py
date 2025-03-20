import hydra
from omegaconf import DictConfig

from server_app import FLServer
import models
import data_preparation
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    """
    Set the initial global model you created in models.py.
    """
    # Build init global model using transformers
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    model_type = cfg.model_type # Check tensorflow or torch model
    model_name = type(model).__name__
    gl_test_torch = models.test_torch() # set torch test    
    
    # Load validation data for evaluating global model
    gl_val_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size) # torch
    
    # Start fl server
    fl_server = FLServer(cfg=cfg, model=model, model_name=model_name, model_type=model_type,
                         gl_val_loader=gl_val_loader, test_torch=gl_test_torch) # torch
    fl_server.start()
    

if __name__ == "__main__":
    main()

