from openvino.inference_engine import IENetwork, IECore
from mo.main import main as openvinomain
from mo.utils.cli_parser import get_onnx_cli_parser


def load_openvino_model(xml_file, bin_file):
    ie = IECore()
    net = IENetwork(model=xml_file, weights=bin_file)
    model = ie.load_network(network=net, device_name="CPU")

    return model


def get_input_shape(xml_file, bin_file):
    """
    Given a model, returns its input shape
    """
    net = IENetwork(model=xml_file, weights=bin_file)
    input_blob = next(iter(net.inputs))
    return net.inputs[input_blob].shape


def async_inference(exec_net, input):
    input_blob = next(iter(exec_net.inputs))
    exec_net.start_async(0, inputs={input_blob: input})
    return exec_net


def get_model_output(exec_net):
    output_blob = next(iter(exec_net.outputs))
    status = exec_net.requests[0].wait(-1)
    if status == 0:
        result = exec_net.requests[0].outputs[output_blob]
        return result