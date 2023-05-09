import numpy as np

from hls4ml.model.hls_layers import FixedPrecisionType, IntegerPrecisionType
from hls4ml.templates import get_supported_boards_dict


class VivadoAcceleratorConfig(object):
    def __init__(self, config, model_inputs, model_outputs):
        self.config = config.config
        self.board = self.config.get('Board', 'pynq-z2')
        boards = get_supported_boards_dict()
        if self.board in boards.keys():
            board_info = boards[self.board]
            self.part = board_info['part']
        else:
            raise Exception('The board does not appear in supported_boards.json file')

        if self.config.get('XilinxPart') is not None:
            if self.config.get('XilinxPart') != self.part:
                print('WARNING: You set a XilinxPart that does not correspond to the Board you specified. The correct '
                      'XilinxPart is now set.')
                self.config['XilinxPart'] = self.part
        accel_config = self.config.get('AcceleratorConfig', None)
        if accel_config is not None:
            prec = accel_config.get('Precision')
            if prec is None:
                raise Exception('Precision must be provided in the AcceleratorConfig')
            else:
                if prec.get('Input') is None or prec.get('Output') is None:
                    raise Exception('Input and Output fields must be provided in the AcceleratorConfig->Precision')
        else:
            accel_config = {'Precision':
                                {
                                    'Input': 'float',
                                    'Output': 'float'
                                },
                            'Driver': 'python',
                            'Interface': 'axi_stream'
                            }
            config.config['AcceleratorConfig'] = accel_config

        self.interface = self.config['AcceleratorConfig'].get('Interface',
                                                              'axi_stream')  # axi_stream, axi_master, axi_lite
        self.driver = self.config['AcceleratorConfig'].get('Driver', 'python')  # python or c
        self.input_type = self.config['AcceleratorConfig']['Precision'].get('Input',
                                                                            'float')  # float, double or ap_fixed<a,b>
        self.output_type = self.config['AcceleratorConfig']['Precision'].get('Output',
                                                                             'float')  # float, double or ap_fixed<a,b>

        assert len(
            model_inputs) == 1, "Only models with one input tensor are currently supported by VivadoAcceleratorBackend"
        assert len(
            model_outputs) == 1, "Only models with one output tensor are currently supported by VivadoAcceleratorBackend"
        self.inp = model_inputs[0]
        self.out = model_outputs[0]
        inp_axi_t = self.input_type
        out_axi_t = self.output_type

        if inp_axi_t not in ['float', 'double']:
            self.input_type = self._next_factor8_type(config.backend.convert_precision_string(inp_axi_t))
        if out_axi_t not in ['float', 'double']:
            self.output_type = self._next_factor8_type(config.backend.convert_precision_string(out_axi_t))

        if inp_axi_t == 'float':
            self.input_bitwidth = 32
        elif out_axi_t == 'double':
            self.input_bitwidth = 64
        else:
            self.input_bitwidth = config.backend.convert_precision_string(inp_axi_t).width

        if out_axi_t == 'float':
            self.output_bitwidth = 32
        elif out_axi_t == 'double':
            self.output_bitwidth = 64
        else:
            self.output_bitwidth = config.backend.convert_precision_string(out_axi_t).width

    def _next_factor8_type(self, p):
        ''' Return a new type with the width rounded to the next factor of 8 up to p's width
            Args:
                p : IntegerPrecisionType or FixedPrecisionType
            Returns:
                An IntegerPrecisionType or FixedPrecisionType with the width rounder up to the next factor of 8
                of p's width. Other parameters (fractional bits, extra modes) stay the same.
        '''
        W = p.width
        newW = int(np.ceil(W / 8) * 8)
        if isinstance(p, FixedPrecisionType):
            return FixedPrecisionType(newW, p.integer, p.signed, p.rounding_mode, p.saturation_mode,
                                      p.saturation_bits)
        elif isinstance(p, IntegerPrecisionType):
            return IntegerPrecisionType(newW, p.signed)

    def get_io_bitwidth(self):
        return self.input_bitwidth, self.output_bitwidth

    def get_corrected_types(self):
        return self.input_type, self.output_type, self.inp, self.out

    def get_interface(self):
        return self.interface

    def get_board_info(self, board=None):
        boards = get_supported_boards_dict()
        if board is None:
            board = self.board
        if board in boards.keys():
            return boards[board]
        else:
            raise Exception('The board is still not supported')

    def get_part(self):
        return self.part

    def get_driver(self):
        return self.driver

    def get_board(self):
        return self.board

    def get_driver_path(self):
        return '../templates/vivado_accelerator/' + self.board + '/' + self.driver + '_drivers/' + \
               self.get_driver_files()

    def get_vivado_ip_wrapper_path(self):
        return '../templates/vivado_accelerator/' + self.board + '/verilog_wrappers'

    def get_vivado_constraints_path(self):
        return '../templates/vivado_accelerator/' + self.board + '/xdc_constraints'

    def get_driver_files(self):
        if self.driver == 'c':
            driver_dir = 'sdk'
            return driver_dir
        elif self.driver == 'python':
            driver_ext = '.py'
            return self.interface + '_driver' + driver_ext

    def get_input_type(self):
        return self.input_type

    def get_output_type(self):
        return self.output_type

    def get_tcl_file_path(self):
        board_info = self.get_board_info(self.board)
        tcl_scripts = board_info.get('tcl_scripts', None)
        if tcl_scripts is None:
            raise Exception('No tcl scripts definition available for the board in supported_board.json')
        tcl_script = tcl_scripts.get(self.interface, None)
        if tcl_script is None:
            raise Exception('No tcl script definition available for the desired interface in supported_board.json')
        return '../templates/vivado_accelerator/' + self.board + '/tcl_scripts/' + tcl_script
