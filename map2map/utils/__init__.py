from .imp import import_attr
from .state import load_model_state_dict

try:
    from .openvino import load_openvino_model, get_input_shape, async_inference, get_model_output, openvinomain, get_onnx_cli_parser
except:
    pass
