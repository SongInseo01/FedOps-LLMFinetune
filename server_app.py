import logging
from typing import Dict, Optional, Tuple
import flwr as fl
import datetime
import os
import json
import numpy as np
import time
import server_api
import server_utils
from collections import OrderedDict
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from models_llm import get_parameters_for_llm, set_parameters_for_llm, set_parameters_for_llm_server

# TF warning log filtering
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

def load_initial_parameters_from_shape(json_path: str):
    """
    parameter_shapes.json 파일을 읽고, 각 shape에 맞는 0으로 초기화된 numpy 파라미터 리스트 반환
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find parameter shape file at: {json_path}")
    
    with open(json_path, "r") as f:
        shape_list = json.load(f)
    
    # numpy 배열 생성 (dtype은 float32가 일반적)
    initial_parameters = [np.zeros(shape, dtype=np.float32) for shape in shape_list]
    return initial_parameters

class FLServer():
    def __init__(self, cfg, model, model_name, model_type, 
                 criterion=None, optimizer=None, scheduler=None, gl_val_loader=None, x_val=None, y_val=None, test_torch=None):
        
        self.task_id = os.environ.get('TASK_ID') # Set FL Task ID

        self.server = server_utils.FLServerStatus() # Set FLServerStatus class
        self.model_type = model_type
        self.strategy = cfg.server.strategy
        
        self.batch_size = int(cfg.batch_size)
        self.local_epochs = int(cfg.num_epochs)
        self.num_rounds = int(cfg.num_rounds)

        self.init_model = model
        self.init_model_name = model_name
        self.next_model = None
        self.next_model_name = None
        
        if self.model_type=="Tensorflow":
            self.x_val = x_val
            self.y_val = y_val  
               

        elif self.model_type == "Pytorch":
            self.gl_val_loader = gl_val_loader
            self.criterion = criterion
            self.optimizer = optimizer
            self.schduler = scheduler
            self.test_torch = test_torch


        elif self.model_type == "Huggingface":
            pass



    def init_gl_model_registration(self, model, gl_model_name) -> None:
        logging.info(f'last_gl_model_v: {self.server.last_gl_model_v}')

        if not model:

            logging.info('init global model making')
            init_model, model_name = self.init_model, self.init_model_name
            print(f'init_gl_model_name: {model_name}')

            self.fl_server_start(init_model, model_name)
            return model_name


        else:
            logging.info('load last global model')
            print(f'last_gl_model_name: {gl_model_name}')

            self.fl_server_start(model, gl_model_name)
            return gl_model_name


    def fl_server_start(self, model, model_name):
        # Load and compile model for
        # 1. server-side parameter initialization
        # 2. server-side parameter evaluation

        model_parameters = None # Init model_parametes variable
        
        if self.model_type == "Tensorflow":
            model_parameters = model.get_weights()
        elif self.model_type == "Pytorch":
            model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        elif self.model_type == "Huggingface":
            json_path = "./parameter_shapes.json"  # 절대경로 또는 상대경로
            model_parameters = load_initial_parameters_from_shape(json_path)


        # if self.model_type == "Huggingface":
        #     from flwrtunellmaggregation_fedavg import FlowerTuneLlm
        #     logging.info("select strategy FlowerTuneLlm")
        #     strategy = FlowerTuneLlm(
        #         fraction_fit=self.strategy.fraction_fit,
        #         fraction_evaluate=self.strategy.fraction_evaluate,
        #         initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        #         on_fit_config_fn=self.fit_config,
        #         on_evaluate_config_fn=self.evaluate_config,

        #     )
        
        # else:
        #     strategy = instantiate(
        #     self.strategy,
        #     initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        #     evaluate_fn=self.get_eval_fn(model, model_name),
        #     on_fit_config_fn=self.fit_config,
        #     on_evaluate_config_fn=self.evaluate_config,
        #     )

        strategy = instantiate(
            self.strategy,
            initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
            evaluate_fn=self.get_eval_fn(model, model_name),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
        
        # Start Flower server (SSL-enabled) for four rounds of federated learning
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )


    def get_eval_fn(self, model, model_name):
        """Return an evaluation function for server-side evaluation."""
        # Load data and model here to avoid the overhead of doing it in `evaluate` itself

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int,
                parameters_ndarrays: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
                        
            # model path for saving local model
            gl_model_path = f'./{model_name}_gl_model_V{self.server.gl_model_v}'
            
            metrics = None
            
            if self.model_type == "Tensorflow":
                # loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
                loss, accuracy = model.evaluate(self.x_val, self.y_val)

                model.set_weights(parameters_ndarrays)  # Update model with the latest parameters
                
                # model save
                model.save(gl_model_path+'.h5')
            
            elif self.model_type == "Pytorch":
                import torch
                keys = [k for k in model.state_dict().keys() if "bn" not in k]
                params_dict = zip(keys, parameters_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
            
                loss, accuracy, metrics = self.test_torch(model, self.gl_val_loader, self.criterion)
                
                # model save
                torch.save(model.state_dict(), gl_model_path+'.pth')

            elif self.model_type == "Huggingface":
                logging.warning("Skipping evaluation for Huggingface model")
                loss, accuracy = 0.0, 0.0

                os.makedirs(gl_model_path, exist_ok=True)
                np.savez(os.path.join(gl_model_path, "adapter_parameters.npz"), *parameters_ndarrays)


                

            if self.server.round >= 1:
                # fit aggregation end time
                self.server.end_by_round = time.time() - self.server.start_by_round
                # gl model performance by round
                if metrics!=None:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_accuracy": accuracy,
                                      "run_time_by_round": self.server.end_by_round, **metrics,"gl_model_v":self.server.gl_model_v}
                else:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round, "gl_loss": loss, "gl_accuracy": accuracy,
                                      "run_time_by_round": self.server.end_by_round,"gl_model_v":self.server.gl_model_v}
                json_server_eval = json.dumps(server_eval_result)
                logging.info(f'server_eval_result - {json_server_eval}')

                # send gl model evaluation to performance pod
                server_api.ServerAPI(self.task_id).put_gl_model_evaluation(json_server_eval)
                
            if metrics!=None:
                return loss, {"accuracy": accuracy, **metrics}
            else:
                return loss, {"accuracy": accuracy}

        return evaluate


    def fit_config(self, rnd: int):
        """Return training configuration dict for each round.
        Keep batch size fixed at 32, perform two rounds of training with one
        local epoch, increase to two local epochs afterwards.
        """
        fl_config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "num_rounds": self.num_rounds,
        }

        # increase round
        self.server.round += 1

        # fit aggregation start time
        self.server.start_by_round = time.time()
        logging.info('server start by round')

        return fl_config


    def evaluate_config(self, rnd: int):
        """Return evaluation configuration dict for each round.
        """
        return {"batch_size": self.batch_size}


    def start(self):

        today_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')


        # Loaded last global model or no global model in s3
        self.next_model, self.next_model_name, self.server.last_gl_model_v = server_utils.model_download_s3(self.task_id, self.model_type, self.init_model)
        
        # Loaded last global model or no global model in local
        # self.next_model, self.next_model_name, self.server.latest_gl_model_v = server_utils.model_download_local(self.model_type, self.init_model)

        # logging.info('Loaded latest global model or no global model')

        # New Global Model Version
        self.server.gl_model_v = self.server.last_gl_model_v + 1

        # API that sends server status to server manager
        inform_Payload = {
            "S3_bucket": "fl-gl-model",  # bucket name
            "Last_GL_Model": "gl_model_%s_V.h5" % self.server.last_gl_model_v,  # Model Weight File Name
            "FLServer_start": today_time,
            "FLSeReady": True,  # server ready status
            "GL_Model_V": self.server.gl_model_v # Current Global Model Version
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)

        try:
            fl_start_time = time.time()

            # Run fl server
            gl_model_name = self.init_gl_model_registration(self.next_model, self.next_model_name)

            fl_end_time = time.time() - fl_start_time  # FL end time

            server_all_time_result = {"fl_task_id": self.task_id, "server_operation_time": fl_end_time,
                                      "gl_model_v": self.server.gl_model_v}
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            
            # Send server time result to performance pod
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
            
            # upload global model
            if self.model_type == "Tensorflow":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.h5"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type =="Pytorch":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type == "Huggingface":
                # 이미 위에서 save된 adapter_parameters.npz 파일을 zip 처리
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}"
                model_save_path = f"./{gl_model_name}_gl_model_V{self.server.gl_model_v}"

                zip_path = f"{model_save_path}.zip"
                server_utils.zip_folder(model_save_path, zip_path)
                server_utils.upload_model_to_bucket(self.task_id, f"{global_model_file_name}.zip")
            
            

            logging.info(f'upload {global_model_file_name} model in s3')

        # server_status error
        except Exception as e:
            logging.error('error: ', e)
            data_inform = {'FLSeReady': False}
            server_api.ServerAPI(self.task_id).put_server_status(json.dumps(data_inform))

        finally:
            logging.info('server close')

            # Modifying the model version in server manager
            server_api.ServerAPI(self.task_id).put_fl_round_fin()
            logging.info('global model version upgrade')