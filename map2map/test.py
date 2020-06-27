from pprint import pprint
import numpy as np
import torch
import sys
import os
from torch.utils.data import DataLoader

from .data import FieldDataset
from . import models
from .models import narrow_like
from .utils import import_attr, load_model_state_dict

try:
    from .utils import load_openvino_model, get_input_shape, async_inference, get_model_output, openvinomain, get_onnx_cli_parser
except:
    pass


def test(args):
    pprint(vars(args))
    sys.stdout.flush()

    test_dataset = FieldDataset(
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        pad=args.pad,
        scale_factor=args.scale_factor,
        cache=args.cache,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batches,
        shuffle=False,
        num_workers=args.loader_workers,
    )

    in_chan, out_chan = test_dataset.in_chan, test_dataset.tgt_chan

    model = import_attr(args.model, models.__name__, args.callback_at)
    model = model(sum(in_chan), sum(out_chan))
    criterion = import_attr(args.criterion, torch.nn.__name__, args.callback_at)
    criterion = criterion()

    if torch.cuda.is_available()==True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(
        state['epoch'], args.load_state))
    del state

    model.eval()

    with torch.no_grad():
        
        # Temporary arrays
        tmp_input_array = []
        tmp_target_array = []
        tmp_output_array = []
        
        for i, (input, target) in enumerate(test_loader):
            
            if i == 0:
                if args.use_openvino == True:
                    if args.openvino_pre_model == True:
                        model = load_openvino_model(args.openvino_xml_file, args.openvino_bin_file)
                        async_net = async_inference(model, input)
                        output = torch.from_numpy(get_model_output(async_net))
                        
                    else:
                        print('I will must trace PyTorch script, convert to ONNX and later to OpenVINO.')
                        print("This will take a while, but don't worry, it's only for now")
                        
                        # Tracing
                        traced_model = torch.jit.trace(model, torch.empty((1, in_chan[0], args.crop+2*args.pad, args.crop+2*args.pad, args.crop+2*args.pad)))
                        
                        # Converting to ONNX
                        torch.onnx.export(traced_model,
                                        torch.empty((1, in_chan[0], args.crop+2*args.pad, args.crop+2*args.pad, args.crop+2*args.pad)),
                                        args.onnx_file,
                                        opset_version=11,
                                        example_outputs=torch.empty((1, out_chan[0], args.crop, args.crop, args.crop)),
                                        verbose=False)
                        
                        # Converting to OpenVINO
                        input_shape = str([1, in_chan[0], args.crop+2*args.pad, args.crop+2*args.pad, args.crop+2*args.pad]).replace(" ", "")
                        path_to_mo_onnx = os.environ['INTEL_OPENVINO_DIR']+'/deployment_tools/model_optimizer/mo_onnx.py'
                        path_to_onnx_model = args.onnx_file
                        sys.argv=[path_to_mo_onnx, "--input_model", path_to_onnx_model, "--input_shape", input_shape, "--data_type", "FP32", "--silent"]
                        
                        openvinomain(get_onnx_cli_parser(), 'onnx')
                        
                        # Inference using OpenVINO
                        model = load_openvino_model(args.openvino_xml_file, args.openvino_bin_file)
                        async_net = async_inference(model, input)
                        output = torch.from_numpy(get_model_output(async_net))
                else:
                    output = model(input)
                    
            else:
                if args.use_openvino == True:
                    async_net = async_inference(model, input)
                    output = torch.from_numpy(get_model_output(async_net))
                else:
                    output = model(input)
            
            
            if args.pad > 0:  # FIXME
                output = narrow_like(output, target)
                input = narrow_like(input, target)
            else:
                target = narrow_like(target, output)
                input = narrow_like(input, output)

            loss = criterion(output, target)

            print('sample {} loss: {}'.format(i, loss.item()))

            if args.in_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.in_norms, np.cumsum(in_chan)):
                    norm(input[:, start:stop], undo=True)
                    start = stop
            if args.tgt_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.tgt_norms, np.cumsum(out_chan)):
                    norm(output[:, start:stop], undo=True)
                    norm(target[:, start:stop], undo=True)
                    start = stop
                    
            tmp_input_array.append(input)
            tmp_target_array.append(target)
            tmp_output_array.append(output)
            
        size = test_dataset.size[0] # This should be unique.
        
        # Assemble arrays:
        input_array = np.empty((in_chan[0],size,size,size))
        output_array = np.empty((out_chan[0],size,size,size))
        target_array = np.empty((out_chan[0],size,size,size))
    
        c=0
        for l in range(0,size,args.crop):
            for m in range(0,size,args.crop):
                for n in range(0,size,args.crop):
                    input_array[:, l:l+args.crop, m:m+args.crop, n:n+args.crop] = tmp_input_array[c][0]
                    output_array[:, l:l+args.crop, m:m+args.crop, n:n+args.crop] = tmp_output_array[c][0]
                    target_array[:, l:l+args.crop, m:m+args.crop, n:n+args.crop] = tmp_target_array[c][0]
                    c+=1

        del tmp_input_array
        del tmp_output_array
        del tmp_target_array
    
    np.savez(f'{args.output}.npz', input=input_array,
            output=output_array, target=target_array)
    
    del input_array
    del output_array
    del target_array